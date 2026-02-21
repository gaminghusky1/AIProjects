import os
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

OUT_DIR = "TransformerData"
MIX_PATH = os.path.join(OUT_DIR, "mixed_owt70_wt30.txt")
SPM_MODEL = os.path.join(OUT_DIR, "spm_openwebtext_wikitext.model")

T = 64              # sequence length for training
VAL_FRACTION = 0.01 # 1% validation
MAX_TOKENS = 5_000_000  # cap for a CPU run; increase later

def count_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n

def iter_text_lines(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line == "<doc>":
                continue
            yield line

def build_packed_examples(sp, lines_iter, T: int, max_tokens: int):
    """
    Yields (x, y) where each is shape (T,).
    We pack tokens across docs, but we insert <eos><bos> boundaries.
    """
    bos = sp.bos_id()
    eos = sp.eos_id()

    buffer = []  # token buffer
    total_tokens = 0

    for line in lines_iter:
        ids = sp.encode(line, out_type=int)
        # doc boundary: <bos> text <eos>
        ids = [bos] + ids + [eos]
        buffer.extend(ids)
        total_tokens += len(ids)
        if total_tokens >= max_tokens:
            break

        # emit as many training windows as possible
        while len(buffer) >= (T + 1):
            chunk = buffer[: T + 1]
            buffer = buffer[T:]  # stride = T (non-overlapping). For more data, use stride < T.
            x = np.array(chunk[:-1], dtype=np.int32)
            y = np.array(chunk[1:],  dtype=np.int32)
            yield x, y

def write_memmaps(x_path, y_path, num_rows, T):
    x_mm = np.memmap(x_path, dtype=np.int32, mode="w+", shape=(num_rows, T))
    y_mm = np.memmap(y_path, dtype=np.int32, mode="w+", shape=(num_rows, T))
    return x_mm, y_mm

def main():
    sp = spm.SentencePieceProcessor()
    sp.Load(SPM_MODEL)

    # Roughly estimate how many sequences we’ll get by a first pass counting tokens
    # (Simple approach: just generate once and store to lists if small; memmap is safer for big.)
    xs = []
    ys = []
    for x, y in tqdm(build_packed_examples(sp, iter_text_lines(MIX_PATH), T, MAX_TOKENS),
                     desc="Tokenizing+packing"):
        xs.append(x)
        ys.append(y)

    xs = np.stack(xs, axis=0)
    ys = np.stack(ys, axis=0)

    # Split train/val
    n = xs.shape[0]
    n_val = max(1, int(n * VAL_FRACTION))
    n_train = n - n_val

    # Shuffle once (important)
    idx = np.random.default_rng(42).permutation(n)
    xs = xs[idx]
    ys = ys[idx]

    x_train, y_train = xs[:n_train], ys[:n_train]
    x_val,   y_val   = xs[n_train:], ys[n_train:]

    np.save(os.path.join(OUT_DIR, f"x_train_T{T}.npy"), x_train)
    np.save(os.path.join(OUT_DIR, f"y_train_T{T}.npy"), y_train)
    np.save(os.path.join(OUT_DIR, f"x_val_T{T}.npy"),   x_val)
    np.save(os.path.join(OUT_DIR, f"y_val_T{T}.npy"),   y_val)

    print("Saved:",
          x_train.shape, y_train.shape,
          x_val.shape, y_val.shape)

if __name__ == "__main__":
    main()