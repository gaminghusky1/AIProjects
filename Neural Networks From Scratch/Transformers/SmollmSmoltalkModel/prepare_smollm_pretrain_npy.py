# prepare_smollm_pretrain_npy.py
#
# Install:
#   pip install sentencepiece tqdm numpy
#
# Run:
#   python prepare_smollm_pretrain_npy.py
#
# Or with custom sequence length:
#   python prepare_smollm_pretrain_npy.py --seq_len 1024

import argparse
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm


DEFAULT_TOKENIZER_PATH = "Tokenizer/smollm_smoltalk_bpe_32k.model"

DEFAULT_TRAIN_TXT = "Data/smollm_train.txt"
DEFAULT_VAL_TXT = "Data/smollm_val.txt"

DEFAULT_OUT_DIR = "Data/tokenized"


def load_tokenizer(path: str):
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp


def choose_dtype(sp):
    vocab_size = sp.vocab_size()

    if vocab_size <= np.iinfo(np.uint16).max:
        return np.uint16

    return np.uint32


def tokenize_file_to_ids(sp, txt_path: str, eos_id: int):
    """
    Tokenizes a text file line by line.

    This does NOT manually add EOS after every line, because your .txt file
    already contains <|endoftext|> between documents.
    """
    ids = []

    txt_path = Path(txt_path)
    file_size = txt_path.stat().st_size

    with txt_path.open("r", encoding="utf-8") as f:
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=f"Tokenizing {txt_path.name}",
        ) as pbar:
            for line in f:
                pbar.update(len(line.encode("utf-8")))

                # Preserve line breaks. Reading line-by-line keeps the newline
                # in `line`, so SentencePiece sees it.
                line_ids = sp.encode(line, out_type=int)
                ids.extend(line_ids)

    return ids


def save_chunks(ids, out_path: str, seq_len: int, dtype):
    chunk_len = seq_len + 1

    usable_len = (len(ids) // chunk_len) * chunk_len
    dropped = len(ids) - usable_len

    if usable_len == 0:
        raise ValueError(
            f"Not enough tokens to create even one chunk of length {chunk_len}."
        )

    ids = np.asarray(ids[:usable_len], dtype=dtype)
    chunks = ids.reshape(-1, chunk_len)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(out_path, chunks)

    return {
        "out_path": str(out_path),
        "num_tokens_total": len(ids) + dropped,
        "num_tokens_used": usable_len,
        "num_tokens_dropped": dropped,
        "num_sequences": chunks.shape[0],
        "chunk_len": chunk_len,
        "shape": chunks.shape,
        "dtype": str(chunks.dtype),
    }


def process_split(sp, txt_path, out_path, seq_len, dtype, eos_id):
    token_ids = tokenize_file_to_ids(sp, txt_path, eos_id=eos_id)
    stats = save_chunks(token_ids, out_path, seq_len=seq_len, dtype=dtype)
    return stats


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--train_txt", type=str, default=DEFAULT_TRAIN_TXT)
    parser.add_argument("--val_txt", type=str, default=DEFAULT_VAL_TXT)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    parser.add_argument("--seq_len", type=int, default=1024)

    args = parser.parse_args()

    tokenizer_path = Path(args.tokenizer)
    train_txt = Path(args.train_txt)
    val_txt = Path(args.val_txt)
    out_dir = Path(args.out_dir)

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    if not train_txt.exists():
        raise FileNotFoundError(f"Train text not found: {train_txt}")

    if not val_txt.exists():
        raise FileNotFoundError(f"Val text not found: {val_txt}")

    sp = load_tokenizer(str(tokenizer_path))

    eos_id = sp.piece_to_id("<|endoftext|>")
    if eos_id < 0:
        raise ValueError("Could not find <|endoftext|> in tokenizer.")

    dtype = choose_dtype(sp)

    print("Tokenizer loaded.")
    print(f"Vocab size: {sp.vocab_size():,}")
    print(f"EOS id: {eos_id}")
    print(f"Using dtype: {dtype}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Saved chunk length: {args.seq_len + 1}")

    train_out = out_dir / f"smollm_train_seq{args.seq_len + 1}.npy"
    val_out = out_dir / f"smollm_val_seq{args.seq_len + 1}.npy"

    train_stats = process_split(
        sp=sp,
        txt_path=str(train_txt),
        out_path=str(train_out),
        seq_len=args.seq_len,
        dtype=dtype,
        eos_id=eos_id,
    )

    val_stats = process_split(
        sp=sp,
        txt_path=str(val_txt),
        out_path=str(val_out),
        seq_len=args.seq_len,
        dtype=dtype,
        eos_id=eos_id,
    )

    print("\nDone.")

    print("\nTrain:")
    for k, v in train_stats.items():
        print(f"  {k}: {v}")

    print("\nVal:")
    for k, v in val_stats.items():
        print(f"  {k}: {v}")

    print("\nTraining usage later:")
    print("  chunks = np.load('Data/tokenized/smollm_train_seq1025.npy', mmap_mode='r')")
    print("  batch = chunks[idx]")
    print("  x = batch[:, :-1]")
    print("  y = batch[:, 1:]")


if __name__ == "__main__":
    main()