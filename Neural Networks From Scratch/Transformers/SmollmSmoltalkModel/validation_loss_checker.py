import math
import time

import numpy as np
import pandas as pd
from sentencepiece import SentencePieceProcessor

from mlx_model import *
import mlx.core as mx

class TokenBatcher:
    def __init__(self, tokens_npy_path: str, batch_size: int):
        self.batch_size = batch_size
        self.tokens = np.load(tokens_npy_path, mmap_mode="r")

        self.num_sequences = self.tokens.shape[0]
        self.chunk_len = self.tokens.shape[1]
        self.seq_len = self.chunk_len - 1

        if self.tokens.ndim != 2:
            raise ValueError(f"Expected tokens to have shape (N, seq_len+1), got {self.tokens.shape}")

    def get_batch(self, batch: int):
        start = batch * self.batch_size
        end = min(start + self.batch_size, self.num_sequences)

        if start >= self.num_sequences:
            raise IndexError(
                f"Batch {batch} starts past the end of the dataset. "
                f"num_batches = {self.num_batches()}"
            )

        batch_tokens = self.tokens[start:end]  # (current_batch_size, seq_len + 1)

        x = batch_tokens[:, :-1].astype(np.int32)
        y = batch_tokens[:, 1:].astype(np.int32)

        return (
            mx.array(x, dtype=mx.int32),
            mx.array(y, dtype=mx.int32),
        )

    def num_batches(self):
        return math.ceil(self.num_sequences / self.batch_size)

def main():
    batcher = TokenBatcher(
        tokens_npy_path="Data/tokenized/smollm_val_seq1025.npy",
        batch_size=8
    )
    smollm_smoltalk_model = model.Model.load_from("Models/smollm_smoltalk_e5_final")

    print("Param count:", smollm_smoltalk_model.get_param_count())
    print("Batch count:", batcher.num_batches())

    test_count = batcher.num_batches()
    running_loss_sum = 0.0
    for i in range(test_count):
        x_test, y_test = batcher.get_batch(i)
        smollm_smoltalk_model.reset_cache()
        curr_loss = smollm_smoltalk_model.test_loss(x_test, y_test)
        running_loss_sum += curr_loss
        print(f"Loss on test {i + 1}: {curr_loss}")

    print(f"Average loss: {running_loss_sum / test_count}")

if __name__ == "__main__":
    main()