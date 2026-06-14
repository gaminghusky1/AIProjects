import math
import time

import numpy as np
import pandas as pd
from sentencepiece import SentencePieceProcessor

from mlx_model import *
import mlx.core as mx

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

        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)

def lr_schedule(step, total_steps, peak_lr, warmup_steps=1000, min_lr_ratio=0.05):
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
            layers.RMSNorm(),
            layers.Attention(d_model, d_model, heads=num_heads, mask=masks.causal, use_rope=True),
        ),
        layers.ResidualBlock(
            layers.RMSNorm(),
            layers.MultilayerPerceptron(d_ff, activation='gelu'),
        ),
    )

def main():
    start_time = time.perf_counter()
    model_name = "useful_model"
    sp = SentencePieceProcessor()
    sp.Load("Tokenizer/useful_spm.model")
    batcher = TokenBatcher(tokens_npy_path="Data/pretrain_train_ids.npy")

    vocab_size = sp.GetPieceSize()
    seq_len = 256
    d_model = 384
    d_ff = 4 * d_model
    num_heads = 6
    num_transformer_blocks = 10

    # embedding_layer = layers.TokenEmbedding(vocab_size, d_model)
    #
    # useful_model = model.Model(
    #     (seq_len,),
    #     embedding_layer,
    #
    #     *[
    #         layer
    #         for _ in range(num_transformer_blocks)
    #         for layer in create_transformer_block(d_model, d_ff, num_heads)
    #     ],
    #
    #     layers.RMSNorm(),
    #     layers.TiedTimeDistributedDense(embedding_layer, activation='linear'),
    # )
    #
    # useful_model.compile(
    #     loss='sparse_softmax_crossentropy',
    #     optimizer='adamw',
    #     optimizer_kwargs={
    #         "weight_decay": 0.1,
    #         "clip_norm": 1.0,
    #         "beta1": 0.9,
    #         "beta2": 0.95,
    #         "epsilon": 1e-8,
    #     },
    # )

    prev_trained_batches = 20000
    end_batch = 22000
    final_end_batch = 40000
    useful_model = model.Model.load_from(f"Models/{model_name}_batch_{prev_trained_batches}")

    print("Param count:", useful_model.get_param_count())

    # metrics = pd.DataFrame(columns=["loss", "ema_loss", "accuracy", "learning_rate"])
    metrics = pd.read_csv(f"Models/{model_name}_metrics.csv", index_col=0)

    # Learning rate schedule
    peak_lr = 2.5e-4

    save_after_batches = 1000
    batches_since_last_save = 0
    ema_loss = metrics.loc[prev_trained_batches]["ema_loss"] if prev_trained_batches in metrics.index else None
    ema_beta = 0.99
    for i in range(prev_trained_batches, end_batch):
        # print(f"Current Batch: {i+1}")
        x_train, y_train = batcher.sample_batch(batch_size=16, seq_len=seq_len)
        lr_curr = lr_schedule(step=i, total_steps=final_end_batch, peak_lr=peak_lr)
        batch_start = time.perf_counter()
        useful_model.fit(x_train, y_train, epochs=1, learning_rate=lr_curr, batch_size=16, verbose=-1)
        batch_time = time.perf_counter() - batch_start
        curr_accuracy = useful_model.get_current_accuracy()
        raw_loss = useful_model.get_current_loss()
        if ema_loss is None:
            ema_loss = raw_loss
        else:
            ema_loss = ema_beta * ema_loss + (1 - ema_beta) * raw_loss
        print(f"Batch {i+1}/{end_batch} finished; Loss: {raw_loss:.5f}, EMA Loss: {ema_loss:.5f}, Accuracy: {curr_accuracy:.5f}, Time: {batch_time:.5f}s")
        metrics.loc[i+1] = {"loss": raw_loss, "ema_loss": ema_loss, "accuracy": curr_accuracy, "learning_rate": lr_curr}
        batches_since_last_save += 1
        if batches_since_last_save >= save_after_batches:
            useful_model.save_as(f"Models/{model_name}_batch_{i+1}")
            metrics.to_csv(f"Models/{model_name}_metrics.csv")
            batches_since_last_save = 0

    # useful_model.save_as(f"Models/{model_name}_model")
    metrics.to_csv(f"Models/{model_name}_metrics.csv")
    print(f"Trained {end_batch - prev_trained_batches} batches in {time.perf_counter() - start_time} seconds.")

if __name__ == "__main__":
    main()