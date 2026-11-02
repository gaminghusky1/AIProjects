# train_sentencepiece_tokenizer.py
#
# Install:
#   pip install sentencepiece tqdm
#
# Expected files:
#   Data/smollm_train.txt
#   Data/smollm_val.txt
#   Data/smoltalk_train.txt
#   Data/smoltalk_val.txt
#
# Output:
#   Data/tokenizer/smollm_smoltalk_bpe_32k.model
#   Data/tokenizer/smollm_smoltalk_bpe_32k.vocab

from pathlib import Path
import argparse
import json
import sentencepiece as spm
from tqdm import tqdm


SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|pad|>",
]


DEFAULT_INPUT_FILES = [
    "Data/smollm_train.txt",
    "Data/smollm_val.txt",
    "Data/smoltalk_train.txt",
    "Data/smoltalk_val.txt",
]


def check_files(paths):
    missing = []
    for p in paths:
        if not Path(p).exists():
            missing.append(p)

    if missing:
        raise FileNotFoundError(
            "Missing input files:\n" + "\n".join(f"  {p}" for p in missing)
        )


def train_tokenizer(
    input_files,
    output_dir,
    model_prefix,
    vocab_size,
    character_coverage,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_prefix_path = str(output_dir / model_prefix)

    input_arg = ",".join(input_files)
    user_symbols = ",".join(SPECIAL_TOKENS)

    spm.SentencePieceTrainer.Train(
        input=input_arg,
        model_prefix=model_prefix_path,

        # Good default for small decoder-only LMs.
        model_type="bpe",
        vocab_size=vocab_size,

        # Critical: no <unk> during training/inference for valid UTF-8 text.
        byte_fallback=True,
        character_coverage=character_coverage,

        # Force chat/control strings to be single pieces.
        user_defined_symbols=user_symbols,

        # IDs. Keep unk even though byte_fallback should prevent real usage.
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=1,
        unk_piece="<unk>",
        pad_piece="<|pad|>",

        # Preserve your already-cleaned text more directly.
        normalization_rule_name="identity",
        remove_extra_whitespaces=False,

        # Helps keep special tags from getting an extra dummy whitespace token.
        add_dummy_prefix=False,

        # Better for code + chat tags + raw LM text.
        split_by_whitespace=True,
        split_digits=True,
        allow_whitespace_only_pieces=True,

        # Needed because byte fallback adds 256 byte tokens plus specials.
        hard_vocab_limit=False,

        # Reasonable for your ~500 MB text.
        input_sentence_size=500_000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
        max_sentence_length=4096,
        num_threads=8,
        minloglevel=1,
    )

    return output_dir / f"{model_prefix}.model"


def load_sp(model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_path))
    return sp


def verify_special_tokens(sp):
    print("\nSpecial token verification:")

    problems = []

    for tok in SPECIAL_TOKENS:
        pieces = sp.encode(tok, out_type=str)
        ids = sp.encode(tok, out_type=int)

        print(f"  {tok}: pieces={pieces}, ids={ids}")

        if len(pieces) != 1 or pieces[0] != tok:
            problems.append((tok, pieces, ids))

    if problems:
        raise RuntimeError(
            "Some special tokens did not encode as exactly one piece:\n"
            + "\n".join(str(p) for p in problems)
        )


def verify_no_unk_on_file(sp, path, max_lines=None):
    unk_id = sp.unk_id()
    total_ids = 0
    unk_count = 0
    line_count = 0

    with open(path, "r", encoding="utf-8") as f:
        iterator = f if max_lines is None else zip(range(max_lines), f)

        if max_lines is None:
            for line in tqdm(iterator, desc=f"Checking {path}"):
                ids = sp.encode(line.rstrip("\n"), out_type=int)
                total_ids += len(ids)
                unk_count += sum(1 for x in ids if x == unk_id)
                line_count += 1
        else:
            for _, line in tqdm(iterator, total=max_lines, desc=f"Checking {path}"):
                ids = sp.encode(line.rstrip("\n"), out_type=int)
                total_ids += len(ids)
                unk_count += sum(1 for x in ids if x == unk_id)
                line_count += 1

    return {
        "file": path,
        "lines": line_count,
        "tokens": total_ids,
        "unk_count": unk_count,
        "unk_rate": unk_count / max(total_ids, 1),
    }


def verify_no_unk(sp, input_files, max_lines_per_file=None):
    print("\nUNK verification:")

    results = []
    total_unk = 0
    total_tokens = 0

    for path in input_files:
        result = verify_no_unk_on_file(
            sp,
            path,
            max_lines=max_lines_per_file,
        )
        results.append(result)

        total_unk += result["unk_count"]
        total_tokens += result["tokens"]

        print(
            f"  {path}: "
            f"tokens={result['tokens']:,}, "
            f"unk={result['unk_count']:,}, "
            f"unk_rate={result['unk_rate']:.8f}"
        )

    if total_unk != 0:
        raise RuntimeError(
            f"Tokenizer produced {total_unk} <unk> tokens over "
            f"{total_tokens:,} checked tokens."
        )

    return results


def save_tokenizer_info(sp, output_dir, model_prefix, verification_results):
    info = {
        "vocab_size": sp.vocab_size(),
        "piece_size": sp.get_piece_size(),
        "unk_id": sp.unk_id(),
        "pad_id": sp.pad_id(),
        "bos_id": sp.bos_id(),
        "eos_id": sp.eos_id(),
        "special_tokens": {
            tok: {
                "id": sp.piece_to_id(tok),
                "pieces_when_encoded_alone": sp.encode(tok, out_type=str),
            }
            for tok in SPECIAL_TOKENS
        },
        "verification": verification_results,
    }

    path = Path(output_dir) / f"{model_prefix}_info.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"\nSaved tokenizer info to {path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_files",
        nargs="+",
        default=DEFAULT_INPUT_FILES,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Data/tokenizer",
    )
    parser.add_argument(
        "--model_prefix",
        type=str,
        default="smollm_smoltalk_bpe_32k",
    )

    # Recommended for your 50M-100M-ish model.
    parser.add_argument("--vocab_size", type=int, default=32_000)

    # 1.0 + byte_fallback is best for "no unknowns".
    parser.add_argument("--character_coverage", type=float, default=1.0)

    # Full check may take a bit. Default checks everything.
    # For a quick test: --max_lines_per_file 10000
    parser.add_argument("--max_lines_per_file", type=int, default=None)

    args = parser.parse_args()

    check_files(args.input_files)

    print("Training SentencePiece tokenizer...")
    print(f"Input files:")
    for p in args.input_files:
        size_mb = Path(p).stat().st_size / (1024 * 1024)
        print(f"  {p} ({size_mb:.2f} MB)")

    model_path = train_tokenizer(
        input_files=args.input_files,
        output_dir=args.output_dir,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
    )

    print(f"\nTokenizer saved to: {model_path}")

    sp = load_sp(model_path)

    print(f"\nFinal vocab size: {sp.vocab_size():,}")
    print(f"UNK id: {sp.unk_id()}")
    print(f"PAD id: {sp.pad_id()}")

    verify_special_tokens(sp)

    verification_results = verify_no_unk(
        sp,
        args.input_files,
        max_lines_per_file=args.max_lines_per_file,
    )

    save_tokenizer_info(
        sp,
        args.output_dir,
        args.model_prefix,
        verification_results,
    )

    print("\nDone. Tokenizer is ready.")


if __name__ == "__main__":
    main()