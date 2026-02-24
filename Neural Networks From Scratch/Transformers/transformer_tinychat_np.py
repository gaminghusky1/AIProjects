import os
import math
import numpy as np
import sentencepiece as spm
import pandas as pd
from datasets import load_dataset
from sentencepiece import SentencePieceProcessor

from base_model import *

class TokenBatcher:
    def __init__(self, tokens_npy_path: str):
        # memory-map so it loads instantly and doesn't duplicate RAM
        self.tokens = np.load(tokens_npy_path, mmap_mode="r")  # shape (N,)

    def sample_batch(self, batch_size: int, seq_len: int):
        N = self.tokens.shape[0]
        if N < seq_len + 2:
            raise ValueError("Token file too small for chosen seq_len.")

        max_start = N - (seq_len + 1)
        starts = np.random.randint(0, max_start, size=batch_size)

        # Build batch in numpy (fast), then convert to MLX
        x = np.empty((batch_size, seq_len), dtype=np.int32)
        y = np.empty((batch_size, seq_len), dtype=np.int32)

        for i, s in enumerate(starts):
            chunk = self.tokens[s : s + seq_len + 1]  # (seq_len+1,)
            x[i] = chunk[:-1]
            y[i] = chunk[1:]

        return x, y

DATASET_NAME = "starhopp3r/TinyChat"
SPECIAL_TOKENS = ["[INST]", "[/INST]"]

def stream_tinychat_text(split="train"):
    ds = load_dataset(DATASET_NAME, split=split, streaming=True)
    for ex in ds:
        txt = ex.get("text", "")
        if txt:
            yield txt

def train_sentencepiece(corpus_txt_path, model_prefix, vocab_size=8000):
    spm.SentencePieceTrainer.Train(
        input=corpus_txt_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        bos_id=1, eos_id=2, pad_id=0, unk_id=3,
        user_defined_symbols=SPECIAL_TOKENS,
    )
    return model_prefix + ".model"

def build_and_save_tokens(
    out_tokens_path="tinychat_tokens.npy",
    tokenizer_dir="./tinychat_tokenizer",
    vocab_size=8000,
    text_limit_for_spm=200_000,
    add_eos_between_rows=True,
):
    os.makedirs(tokenizer_dir, exist_ok=True)
    corpus_path = os.path.join(tokenizer_dir, "corpus.txt")
    model_prefix = os.path.join(tokenizer_dir, "spm")

    # 1) write a corpus for SPM (can be a subset)
    with open(corpus_path, "w", encoding="utf-8") as f:
        for i, txt in enumerate(stream_tinychat_text()):
            f.write(txt.replace("\n", " ") + "\n")
            if text_limit_for_spm is not None and (i + 1) >= text_limit_for_spm:
                break

    # 2) train tokenizer
    model_path = train_sentencepiece(corpus_path, model_prefix, vocab_size=vocab_size)
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    eos = sp.eos_id()

    # 3) tokenize full dataset (streaming)
    ids = []
    for txt in stream_tinychat_text():
        ids.extend(sp.EncodeAsIds(txt))
        if add_eos_between_rows and eos != -1:
            ids.append(eos)

    ids = np.asarray(ids, dtype=np.int32)

    # 4) save token stream fast
    np.save(out_tokens_path, ids)  # uncompressed, fast
    return out_tokens_path, model_path

def lr_schedule(step, total_steps, peak_lr, warmup_steps=500, min_lr_ratio=0.1):
    step = min(step, total_steps - 1)
    # Linear warmup
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps

    # Cosine decay to min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = peak_lr * min_lr_ratio
    return min_lr + (peak_lr - min_lr) * cosine

def create_transformer_block(d_model, d_ff, num_heads):
    return (
        layers.ResidualBlock(
            layers.LayerNorm(),
            layers.Attention(d_model, d_model, heads=num_heads, mask=masks.causal),
        ),
        layers.ResidualBlock(
            layers.LayerNorm(),
            layers.MultilayerPerceptron(d_ff, activation='gelu'),
        ),
    )

def main():
    # print(build_and_save_tokens())
    sp = SentencePieceProcessor()
    sp.Load("tinychat_tokenizer/spm.model")
    batcher = TokenBatcher(tokens_npy_path="tinychat_tokens.npy")

    vocab_size = sp.GetPieceSize()
    seq_len = 512
    d_model = 384
    d_ff = 4 * d_model
    num_heads = 6
    num_transformer_blocks = 10

    tinychat_model = model.Model(
        (seq_len,),
        layers.TokenEmbedding(vocab_size, d_model),
        layers.PositionalEmbedding(),

        *[
            layer
            for _ in range(num_transformer_blocks)
            for layer in create_transformer_block(d_model, d_ff, num_heads)
        ],

        layers.LayerNorm(),
        layers.TimeDistributedDense(vocab_size, activation='crossentropy_softmax'),
    )

    tinychat_model.compile(loss='softmax_crossentropy', optimizer='adam')

    start_batch = 0
    end_batch = 1000
    final_end_batch = 20000
    # tinychat_model = model.Model.load_from(f"TinychatModels/better_tinychat_model_batch_{start_batch}")

    print("Param count:", tinychat_model.get_param_count())

    metrics = pd.DataFrame(columns=["loss", "ema_loss", "accuracy"])
    # metrics = pd.read_csv("TinychatModels/better_tinychat_metrics.csv", index_col=0)

    # Learning rate schedule
    peak_lr = 2e-3

    save_after_batches = 500
    batches_since_last_save = 0
    ema_loss = None
    ema_beta = 0.99
    for i in range(start_batch, end_batch):
        # print(f"Current Batch: {i+1}")
        x_train, y_train = batcher.sample_batch(batch_size=16, seq_len=seq_len)
        lr_curr = lr_schedule(i, final_end_batch, peak_lr)
        tinychat_model.fit(x_train, y_train, epochs=1, learning_rate=lr_curr, batch_size=16, verbose=-1, y_ohe=False)
        curr_accuracy = tinychat_model.get_current_accuracy()
        raw_loss = tinychat_model.get_current_loss()
        if ema_loss is None:
            ema_loss = raw_loss
        else:
            ema_loss = ema_beta * ema_loss + (1 - ema_beta) * raw_loss
        print(f"Batch {i+1}/{end_batch} finished; Loss: {raw_loss:.5f}, EMA Loss: {ema_loss:.5f}, Accuracy: {curr_accuracy:.5f}")
        metrics.loc[i+1] = {"loss": raw_loss, "ema_loss": ema_loss, "accuracy": curr_accuracy}
        batches_since_last_save += 1
        if batches_since_last_save >= save_after_batches:
            tinychat_model.save_as(f"TinychatModels/better_tinychat_model_batch_{i+1}")
            metrics.to_csv("TinychatModels/better_tinychat_metrics.csv")
            batches_since_last_save = 0

    # tinychat_model.save_as("TinychatModels/better_tinychat_model")
    metrics.to_csv("TinychatModels/better_tinychat_metrics.csv")

if __name__ == "__main__":
    main()