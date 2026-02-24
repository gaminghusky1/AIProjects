import os
import re
import math
import numpy as np
import sentencepiece as spm
from datasets import load_dataset
import mlx.core as mx
from mlx_model import *
import pandas as pd

DATASET_NAME = "starhopp3r/TinyChat"
TEXT_FIELD = "text"
SPECIAL_TOKENS = ["[INST]", "[/INST]"]

def _stream_conversations(split="train"):
    ds = load_dataset(DATASET_NAME, split=split, streaming=True)
    for ex in ds:
        txt = ex.get(TEXT_FIELD, "")
        if txt:
            yield txt

def _normalize_conversation(txt: str) -> str:
    txt = re.sub(r"\s+", " ", txt.strip())
    if txt.startswith("[/INST]"):
        txt = "[INST] " + txt
    return txt

def _train_sentencepiece(corpus_txt_path: str, model_prefix: str, vocab_size: int):
    spm.SentencePieceTrainer.Train(
        input=corpus_txt_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        bos_id=1, eos_id=2, pad_id=0, unk_id=3,
        user_defined_symbols=",".join(SPECIAL_TOKENS),
        hard_vocab_limit=False,
        character_coverage=1.0,
    )
    return model_prefix + ".model"


def build_tinychat_npz(
    out_npz_path: str = "tinychat_conversations_eos.npz",
    tokenizer_dir: str = "./tinychat_tokenizer",
    vocab_size: int = 8000,
    spm_corpus_rows: int = 200_000,
    split: str = "train",
    max_conversations: int | None = None,
    add_eos: bool = True,
):
    """
    Builds SPM tokenizer, tokenizes each dataset row as one conversation,
    optionally appends eos_id to each conversation, and saves as NPZ with
    a compact (tokens, offsets, lengths) representation.
    """
    os.makedirs(tokenizer_dir, exist_ok=True)
    corpus_path = os.path.join(tokenizer_dir, "corpus.txt")
    model_prefix = os.path.join(tokenizer_dir, "spm")

    # 1) write corpus: one conversation per line
    with open(corpus_path, "w", encoding="utf-8") as f:
        for i, raw in enumerate(_stream_conversations(split=split)):
            txt = _normalize_conversation(raw)
            if txt:
                f.write(txt + "\n")
            if spm_corpus_rows is not None and (i + 1) >= spm_corpus_rows:
                break

    # 2) train/load tokenizer
    model_path = _train_sentencepiece(corpus_path, model_prefix, vocab_size=vocab_size)
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)

    eos_id = sp.eos_id()
    inst_id = sp.PieceToId("[INST]")
    close_id = sp.PieceToId("[/INST]")

    if inst_id < 0 or close_id < 0:
        raise RuntimeError("SPM did not register [INST]/[/INST] as single pieces. Check trainer args.")
    if add_eos and eos_id < 0:
        raise RuntimeError("Tokenizer has no eos_id but add_eos=True.")

    # 3) tokenize each conversation separately -> compact storage
    offsets = []
    lengths = []
    chunks = []
    cursor = 0
    skipped = 0
    n_conv = 0

    for raw in _stream_conversations(split=split):
        if max_conversations is not None and n_conv >= max_conversations:
            break

        txt = _normalize_conversation(raw)
        if not txt or "[INST]" not in txt:
            skipped += 1
            continue

        ids = sp.EncodeAsIds(txt)
        if add_eos:
            ids.append(eos_id)

        if len(ids) < 2:
            skipped += 1
            continue

        offsets.append(cursor)
        lengths.append(len(ids))
        arr = np.asarray(ids, dtype=np.int32)
        chunks.append(arr)
        cursor += arr.size
        n_conv += 1

    tokens = np.concatenate(chunks, axis=0) if chunks else np.zeros((0,), dtype=np.int32)
    offsets = np.asarray(offsets, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.int32)

    np.savez(
        out_npz_path,
        tokens=tokens,
        offsets=offsets,
        lengths=lengths,
        vocab_size=np.asarray([vocab_size], dtype=np.int32),
        eos_id=np.asarray([eos_id], dtype=np.int32),
        inst_id=np.asarray([inst_id], dtype=np.int32),
        close_id=np.asarray([close_id], dtype=np.int32),
        dataset_name=np.asarray([DATASET_NAME]),
        split=np.asarray([split]),
        skipped=np.asarray([skipped], dtype=np.int32),
        spm_model_path=np.asarray([model_path]),
    )
    return out_npz_path, model_path


def load_xy_from_npz(npz_path: str):
    """
    Loads NPZ and returns:
      x_train: list[mx.array] where x_train[i] = conv_tokens[:-1]
      y_train: list[mx.array] where y_train[i] = conv_tokens[1:]
    """
    z = np.load(npz_path)
    tokens = z["tokens"].astype(np.int32, copy=False)
    offsets = z["offsets"].astype(np.int64, copy=False)
    lengths = z["lengths"].astype(np.int32, copy=False)

    x_train = []
    y_train = []

    for i in range(offsets.shape[0]):
        start = int(offsets[i])
        L = int(lengths[i])
        conv = tokens[start:start + L]  # includes eos if you added it

        # x = all but last token, y = all but first token
        x_np = conv[:-1]
        y_np = conv[1:]

        x_train.append(mx.array(x_np, dtype=mx.int32))
        y_train.append(mx.array(y_np, dtype=mx.int32))

    return x_train, y_train

def load_xy_numpy_views_from_npz(npz_path: str):
    z = np.load(npz_path)
    tokens = z["tokens"].astype(np.int32, copy=False)
    offsets = z["offsets"].astype(np.int64, copy=False)
    lengths = z["lengths"].astype(np.int32, copy=False)

    x_views = []
    y_views = []
    for i in range(offsets.shape[0]):
        start = int(offsets[i])
        L = int(lengths[i])
        conv = tokens[start:start + L]
        x_views.append(conv[:-1])
        y_views.append(conv[1:])
    return x_views, y_views

def to_one_hot(indices, vocab_size):
    y = np.zeros((len(indices), vocab_size), dtype=np.float32)
    y[np.arange(len(indices)), indices] = 1
    return y

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
    x_view, y_view = load_xy_numpy_views_from_npz("tinychat_per_conversation_tokens.npz")
    sp = spm.SentencePieceProcessor()
    sp.Load("tinychat_tokenizer/spm.model")

    vocab_size = sp.GetPieceSize()
    d_model = 384
    d_ff = 4 * d_model
    num_heads = 6
    num_transformer_blocks = 10

    tinychat_model = model.Model(
        (-1,),
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

    # tinychat_model = model.Model.load_from(f"TinychatModels/per_conversation_tinychat_model_batch_{start_batch}")

    print("Param count:", tinychat_model.get_param_count())

    metrics = pd.DataFrame(columns=["loss", "ema_loss", "accuracy"])
    # metrics = pd.read_csv("TinychatModels/per_conversation_tinychat_metrics.csv", index_col=0)

    # Learning rate schedule
    # peak_lr = 2e-3

    start_conversation = 0
    end_conversation = 20000
    # final_end_conversation = 500000

    save_after_conversations = 10000
    conversations_since_last_save = 0
    conversation_group_size = 2000
    ema_loss = None
    ema_beta = 0.99
    for i in range(start_conversation, end_conversation, conversation_group_size):
        # lr_curr = lr_schedule(i, final_end_conversation, peak_lr)
        tinychat_model.fit([mx.array(x_view[x]) for x in range(i, i + conversation_group_size)], [mx.array(y_view[y]) for y in range(i, i + conversation_group_size)], epochs=1, learning_rate=0.005, batch_size=1, verbose=2, y_ohe=False, shuffle=False)
        curr_accuracy = tinychat_model.get_current_accuracy()
        raw_loss = tinychat_model.get_current_loss()
        if ema_loss is None:
            ema_loss = raw_loss
        else:
            ema_loss = ema_beta * ema_loss + (1 - ema_beta) * raw_loss
        print(f"Conversation {i+1}/{end_conversation} finished; Loss: {raw_loss:.5f}, EMA Loss: {ema_loss:.5f}, Accuracy: {curr_accuracy:.5f}")
        metrics.loc[i+1] = {"loss": raw_loss, "ema_loss": ema_loss, "accuracy": curr_accuracy}
        conversations_since_last_save += 1
        if conversations_since_last_save >= save_after_conversations:
            tinychat_model.save_as(f"TinychatModels/per_conversation_tinychat_model_conversation_{i+1}")
            metrics.to_csv("TinychatModels/per_conversation_tinychat_metrics.csv")
            conversations_since_last_save = 0

    # tinychat_model.save_as("TinychatModels/per_conversation_tinychat_model")
    metrics.to_csv("TinychatModels/per_conversation_tinychat_metrics.csv")

# npz_path, spm_model_path = build_tinychat_npz(
#     out_npz_path="tinychat_per_conversation_tokens.npz",
#     vocab_size=8000,
#     split="train",
#     add_eos=True,
# )
#
# print(npz_path, spm_model_path)
if __name__ == "__main__":
    main()
