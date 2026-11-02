"""
Train a SentencePiece tokenizer on the UsefulModel corpus and tokenize it.

Default inputs:
    Data/pretrain_train.txt
    Data/pretrain_val.txt
    Data/sft_train.txt
    Data/sft_val.txt

Default outputs:
    Tokenizer/useful_spm.model
    Tokenizer/useful_spm.vocab
    Tokenizer/tokenizer_meta.json
    Data/pretrain_train_ids.npy
    Data/pretrain_val_ids.npy
    Data/sft_train_ids.npy
    Data/sft_val_ids.npy

The tokenizer is intentionally trained on all available generated text files.
Special formatting tokens are user-defined symbols, so they encode as single
pieces instead of being split into smaller subwords.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import sentencepiece as spm


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "Data"
DEFAULT_TOKENIZER_DIR = BASE_DIR / "Tokenizer"
DEFAULT_MODEL_PREFIX = "useful_spm"

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_META_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|document|>",
    "[INST]",
    "[/INST]",
    "<<SYS>>",
    "<</SYS>>",
]

DEFAULT_TEXT_FILES = [
    "pretrain_train.txt",
    "pretrain_val.txt",
    "sft_train.txt",
    "sft_val.txt",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def existing_files(paths: list[Path]) -> list[Path]:
    found = [path for path in paths if path.exists() and path.stat().st_size > 0]
    missing = [str(path) for path in paths if path not in found]
    if missing:
        print("Skipping missing/empty files:")
        for path in missing:
            print(f"  {path}")
    return found


def train_sentencepiece(
    *,
    input_files: list[Path],
    tokenizer_dir: Path,
    model_prefix: str,
    vocab_size: int,
    model_type: str,
    character_coverage: float,
    max_sentence_length: int,
    num_threads: int,
    overwrite: bool,
) -> tuple[Path, bool]:
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    model_path = tokenizer_dir / f"{model_prefix}.model"

    if model_path.exists() and not overwrite:
        print(f"Tokenizer already exists: {model_path}")
        return model_path, False

    input_arg = ",".join(str(path) for path in input_files)
    prefix_arg = str(tokenizer_dir / model_prefix)

    print("Training SentencePiece on:")
    for path in input_files:
        print(f"  {path}")

    spm.SentencePieceTrainer.Train(
        input=input_arg,
        model_prefix=prefix_arg,
        model_type=model_type,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        input_sentence_size=0,
        shuffle_input_sentence=False,
        train_extremely_large_corpus=True,
        max_sentence_length=max_sentence_length,
        num_threads=num_threads,
        hard_vocab_limit=False,
        byte_fallback=True,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece=PAD_TOKEN,
        bos_piece=BOS_TOKEN,
        eos_piece=EOS_META_TOKEN,
        unk_piece=UNK_TOKEN,
        user_defined_symbols=",".join(SPECIAL_TOKENS),
    )

    return model_path, True


def load_spm(model_path: Path) -> spm.SentencePieceProcessor:
    sp = spm.SentencePieceProcessor()
    if not sp.Load(str(model_path)):
        raise RuntimeError(f"Failed to load SentencePiece model: {model_path}")
    return sp


def assert_special_tokens_are_single_pieces(sp: spm.SentencePieceProcessor):
    failures = []
    for token in SPECIAL_TOKENS:
        token_id = sp.PieceToId(token)
        ids = sp.EncodeAsIds(token)
        pieces = sp.EncodeAsPieces(token)
        if token_id == sp.unk_id() or ids != [token_id]:
            failures.append(
                {
                    "token": token,
                    "piece_id": int(token_id),
                    "encoded_ids": [int(i) for i in ids],
                    "encoded_pieces": pieces,
                }
            )

    if failures:
        raise RuntimeError(
            "Some required special tokens are not single SentencePiece pieces:\n"
            + json.dumps(failures, indent=2)
        )


def train_or_retrain_sentencepiece(args: argparse.Namespace, input_files: list[Path]) -> tuple[Path, spm.SentencePieceProcessor, bool]:
    model_path, trained_now = train_sentencepiece(
        input_files=input_files,
        tokenizer_dir=args.tokenizer_dir,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        max_sentence_length=args.max_sentence_length,
        num_threads=args.num_threads,
        overwrite=args.overwrite_tokenizer,
    )

    sp = load_spm(model_path)
    try:
        assert_special_tokens_are_single_pieces(sp)
    except RuntimeError:
        if trained_now or args.overwrite_tokenizer:
            raise

        print(
            "Existing tokenizer is incompatible with the required special-token "
            "encoding. Retraining it with add_dummy_prefix=False."
        )
        model_path, trained_now = train_sentencepiece(
            input_files=input_files,
            tokenizer_dir=args.tokenizer_dir,
            model_prefix=args.model_prefix,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage,
            max_sentence_length=args.max_sentence_length,
            num_threads=args.num_threads,
            overwrite=True,
        )
        sp = load_spm(model_path)
        assert_special_tokens_are_single_pieces(sp)

    return model_path, sp, trained_now


def count_encoded_tokens(sp: spm.SentencePieceProcessor, text_path: Path) -> int:
    total = 0
    with text_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line:
                total += len(sp.EncodeAsIds(line.rstrip("\n")))
    return total


def fill_token_memmap(
    sp: spm.SentencePieceProcessor,
    *,
    text_path: Path,
    out_path: Path,
    token_count: int,
):
    arr = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.int32,
        shape=(token_count,),
    )

    idx = 0
    with text_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line:
                continue

            ids = sp.EncodeAsIds(line.rstrip("\n"))
            if not ids:
                continue

            next_idx = idx + len(ids)
            arr[idx:next_idx] = np.asarray(ids, dtype=np.int32)
            idx = next_idx

    if idx != token_count:
        raise RuntimeError(
            f"Token count changed while writing {out_path}: expected {token_count}, got {idx}"
        )

    arr.flush()


def tokenize_file(
    sp: spm.SentencePieceProcessor,
    *,
    text_path: Path,
    out_path: Path,
    overwrite: bool,
) -> dict:
    if out_path.exists() and not overwrite:
        print(f"Token file already exists: {out_path}")
        return {
            "text_path": str(text_path),
            "ids_path": str(out_path),
            "token_count": np.load(out_path, mmap_mode="r").shape[0],
            "skipped_existing": True,
        }

    print(f"Counting tokens: {text_path}")
    token_count = count_encoded_tokens(sp, text_path)
    print(f"Writing {token_count:,} tokens to {out_path}")
    fill_token_memmap(sp, text_path=text_path, out_path=out_path, token_count=token_count)

    return {
        "text_path": str(text_path),
        "ids_path": str(out_path),
        "token_count": token_count,
        "skipped_existing": False,
    }


def write_metadata(
    *,
    tokenizer_dir: Path,
    model_path: Path,
    input_files: list[Path],
    tokenized_files: list[dict],
    args: argparse.Namespace,
    sp: spm.SentencePieceProcessor,
):
    special_token_ids = {token: int(sp.PieceToId(token)) for token in SPECIAL_TOKENS}
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "vocab_size": int(sp.GetPieceSize()),
        "model_type": args.model_type,
        "byte_fallback": True,
        "normalization_rule_name": "identity",
        "add_dummy_prefix": False,
        "meta_tokens": {
            "pad": {"piece": PAD_TOKEN, "id": int(sp.pad_id())},
            "bos": {"piece": BOS_TOKEN, "id": int(sp.bos_id())},
            "eos": {"piece": EOS_META_TOKEN, "id": int(sp.eos_id())},
            "unk": {"piece": UNK_TOKEN, "id": int(sp.unk_id())},
        },
        "special_token_ids": special_token_ids,
        "input_files": [str(path) for path in input_files],
        "tokenized_files": tokenized_files,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }

    meta_path = tokenizer_dir / "tokenizer_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote metadata: {meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SentencePiece and tokenize UsefulModel data.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--model-prefix", default=DEFAULT_MODEL_PREFIX)
    parser.add_argument("--vocab-size", type=int, default=12_000)
    parser.add_argument("--model-type", choices=["bpe", "unigram"], default="bpe")
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    parser.add_argument("--max-sentence-length", type=int, default=65_536)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--overwrite-tokenizer", action="store_true")
    parser.add_argument("--overwrite-ids", action="store_true")
    parser.add_argument(
        "--files",
        nargs="*",
        default=DEFAULT_TEXT_FILES,
        help="Text files under --data-dir to train on and tokenize.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    requested_files = [args.data_dir / name for name in args.files]
    input_files = existing_files(requested_files)
    if not input_files:
        raise FileNotFoundError(f"No non-empty text files found under {args.data_dir}")

    model_path, sp, trained_now = train_or_retrain_sentencepiece(args, input_files)
    overwrite_ids = args.overwrite_ids or trained_now

    tokenized_files = []
    for text_path in input_files:
        out_path = text_path.with_name(f"{text_path.stem}_ids.npy")
        tokenized_files.append(
            tokenize_file(
                sp,
                text_path=text_path,
                out_path=out_path,
                overwrite=overwrite_ids,
            )
        )

    write_metadata(
        tokenizer_dir=args.tokenizer_dir,
        model_path=model_path,
        input_files=input_files,
        tokenized_files=tokenized_files,
        args=args,
        sp=sp,
    )

    print("Done.")


if __name__ == "__main__":
    main()
