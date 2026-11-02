import math
import time
import os

import numpy as np
import pandas as pd
from sentencepiece import SentencePieceProcessor

from mlx_model import *
import mlx.core as mx

class TokenBatcher:
    def __init__(self, tokens_npy_path: str, index_order_path: str, batch_size: int):
        self.batch_size = batch_size
        self.tokens = np.load(tokens_npy_path, mmap_mode="r")

        self.num_sequences = self.tokens.shape[0]
        self.chunk_len = self.tokens.shape[1]
        self.seq_len = self.chunk_len - 1

        self.index_order = np.load(index_order_path, mmap_mode="r")

        if self.tokens.ndim != 2:
            raise ValueError(f"Expected tokens to have shape (N, seq_len+1), got {self.tokens.shape}")

        if self.index_order.shape[0] != self.num_sequences:
            raise ValueError(
                f"Index order length {self.index_order.shape[0]} does not match "
                f"num_sequences {self.num_sequences}"
            )

    def get_batch(self, batch: int):
        start = batch * self.batch_size
        end = min(start + self.batch_size, self.num_sequences)

        if start >= self.num_sequences:
            raise IndexError(
                f"Batch {batch} starts past the end of the dataset. "
                f"num_batches = {self.num_batches()}"
            )

        indices = self.index_order[start:end]
        current_batch_size = len(indices)

        batch_tokens = self.tokens[indices]  # (current_batch_size, seq_len + 1)

        x = batch_tokens[:, :-1].astype(np.int32)
        y = batch_tokens[:, 1:].astype(np.int32)

        return (
            current_batch_size,
            mx.array(x, dtype=mx.int32),
            mx.array(y, dtype=mx.int32),
        )

    def num_batches(self):
        return math.ceil(self.num_sequences / self.batch_size)

def lr_schedule(step, total_steps, peak_lr=3e-4, warmup_steps=200, min_lr=3e-5):
    step = min(step, total_steps - 1)
    # Linear warmup
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps

    # Cosine decay to min_lr
    decay_steps = max(1, total_steps - warmup_steps - 1)
    decay_step = step - warmup_steps
    progress = min(1.0, decay_step / decay_steps)

    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + cosine * (peak_lr - min_lr)

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
    model_name = "smollm_smoltalk"
    sp = SentencePieceProcessor()
    sp.Load("Tokenizer/smollm_smoltalk_bpe_32k.model")

    curr_epoch = 5
    total_epochs = 5
    prev_trained_batches = 10000
    this_run_end_batch = 12000

    batcher = TokenBatcher(
        tokens_npy_path="Data/tokenized/smollm_train_seq1025.npy",
        index_order_path=f"Data/tokenized/smollm_train_order_e{curr_epoch}.npy",
        batch_size=8
    )

    this_run_end_batch = min(this_run_end_batch, batcher.num_batches())

    save_frequency_batches = 500

    seq_len = 1024

    if seq_len != batcher.seq_len:
        raise ValueError(f"Model seq_len {seq_len} != data seq_len {batcher.seq_len}")

    # d_model = 512
    # d_ff = 4 * d_model
    # num_heads = 8
    # num_transformer_blocks = 12
    #
    #
    # vocab_size = sp.GetPieceSize()
    # embedding_layer = layers.TokenEmbedding(vocab_size, d_model)
    #
    # smollm_smoltalk_model = model.Model(
    #     (None,),
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
    # smollm_smoltalk_model.compile(
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

    # Learning rate schedule
    peak_lr = 3e-4
    min_lr = 3e-5
    warmup_steps = 300
    total_steps = total_epochs * batcher.num_batches()

    prev_epoch_steps = batcher.num_batches() * (curr_epoch - 1)
    try:
        smollm_smoltalk_model = model.Model.load_from(f"Models/{model_name}_e{curr_epoch}_batch_{prev_trained_batches}")
        print(f"Loading from epoch {curr_epoch} batch {prev_trained_batches}...")
    except FileNotFoundError:
        smollm_smoltalk_model = model.Model.load_from(f"Models/{model_name}_e{curr_epoch - 1}_final")
        print(f"Loading from epoch {curr_epoch - 1} final...")

    print("Param count:", smollm_smoltalk_model.get_param_count())
    print("Total batches per epoch:", batcher.num_batches())

    metrics_path = f"Models/{model_name}_metrics.csv"
    if os.path.exists(metrics_path):
        print("Found existing metrics file:", metrics_path)
        metrics = pd.read_csv(metrics_path, index_col=0)
        metrics.index = metrics.index.astype(int)
    else:
        metrics = pd.DataFrame(columns=["loss", "ema_loss", "accuracy", "learning_rate"])

    resume_step = prev_trained_batches + prev_epoch_steps
    ema_loss = metrics.loc[resume_step]["ema_loss"] if resume_step in metrics.index else None
    ema_beta = 0.99
    for i in range(prev_trained_batches, this_run_end_batch):
        curr_batch_size, x_train, y_train = batcher.get_batch(i)
        lr_curr = lr_schedule(step=i+prev_epoch_steps, total_steps=total_steps, peak_lr=peak_lr, warmup_steps=warmup_steps, min_lr=min_lr)
        batch_start = time.perf_counter()
        smollm_smoltalk_model.fit(x_train, y_train, epochs=1, learning_rate=lr_curr, batch_size=curr_batch_size, verbose=-1)
        batch_time = time.perf_counter() - batch_start
        curr_accuracy = smollm_smoltalk_model.get_current_accuracy()
        raw_loss = smollm_smoltalk_model.get_current_loss()
        if ema_loss is None:
            ema_loss = raw_loss
        else:
            ema_loss = ema_beta * ema_loss + (1 - ema_beta) * raw_loss
        print(f"Epoch {curr_epoch}/{total_epochs}; Batch {i+1}/{this_run_end_batch}; Loss: {raw_loss:.5f}, EMA Loss: {ema_loss:.5f}, Accuracy: {curr_accuracy:.5f}, Time: {batch_time:.5f}s")
        metrics.loc[i+1 + prev_epoch_steps] = {"loss": raw_loss, "ema_loss": ema_loss, "accuracy": curr_accuracy, "learning_rate": lr_curr}
        if (i + 1) % save_frequency_batches == 0:
            smollm_smoltalk_model.save_as(f"Models/{model_name}_e{curr_epoch}_batch_{i+1}")
            metrics.to_csv(metrics_path)

    if this_run_end_batch == batcher.num_batches():
        smollm_smoltalk_model.save_as(f"Models/{model_name}_e{curr_epoch}_final")

    metrics.to_csv(metrics_path)
    print(f"Trained {this_run_end_batch - prev_trained_batches} batches in {time.perf_counter() - start_time} seconds.")

if __name__ == "__main__":
    main()