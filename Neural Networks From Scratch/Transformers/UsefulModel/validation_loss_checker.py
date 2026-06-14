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

def main():
    batcher = TokenBatcher(tokens_npy_path="Data/pretrain_val_ids.npy")
    useful_model = model.Model.load_from(f"Models/useful_model_batch_20000")

    print("Param count:", useful_model.get_param_count())

    test_count = 100
    running_loss_sum = 0.0
    for i in range(test_count):
        x_test, y_test = batcher.sample_batch(batch_size=16, seq_len=256)
        curr_loss = useful_model.test_loss(x_test, y_test)
        running_loss_sum += curr_loss
        print(f"Loss on test {i + 1}: {curr_loss}")

    print(f"Average loss: {running_loss_sum / test_count}")

if __name__ == "__main__":
    main()