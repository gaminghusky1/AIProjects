# prepare_smoltalk_sft_npy.py
#
# Install:
#   pip install sentencepiece tqdm numpy
#
# Run:
#   python prepare_smoltalk_sft_npy.py
#
# Optional:
#   python prepare_smoltalk_sft_npy.py --seq_len 1024

import argparse
import re
from pathlib import Path

import numpy as np
import sentencepiece as spm
from tqdm import tqdm


DEFAULT_TOKENIZER_PATH = "Tokenizer/smollm_smoltalk_bpe_32k.model"

DEFAULT_TRAIN_TXT = "Data/smoltalk_train.txt"
DEFAULT_VAL_TXT = "Data/smoltalk_val.txt"

DEFAULT_OUT_DIR = "Data/tokenized"

SYSTEM_TAG = "<|system|>"
USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"
EOS_TAG = "<|endoftext|>"

TAGS = [SYSTEM_TAG, USER_TAG, ASSISTANT_TAG, EOS_TAG]
TAG_PATTERN = re.compile(
    "(" + "|".join(re.escape(tag) for tag in TAGS) + ")"
)


def load_tokenizer(path: str):
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp


def choose_token_dtype(sp):
    if sp.vocab_size() <= np.iinfo(np.uint16).max:
        return np.uint16
    return np.uint32


def get_special_ids(sp):
    ids = {
        "system": sp.piece_to_id(SYSTEM_TAG),
        "user": sp.piece_to_id(USER_TAG),
        "assistant": sp.piece_to_id(ASSISTANT_TAG),
        "eos": sp.piece_to_id(EOS_TAG),
    }

    for name, token_id in ids.items():
        if token_id < 0:
            raise ValueError(f"Tokenizer is missing special token: {name}")

    return ids


def verify_special_tokens_are_single(sp):
    for tag in TAGS:
        pieces = sp.encode(tag, out_type=str)
        ids = sp.encode(tag, out_type=int)

        if len(ids) != 1 or pieces[0] != tag:
            raise ValueError(
                f"{tag} is not encoding as one token.\n"
                f"pieces={pieces}\n"
                f"ids={ids}"
            )


def encode_segment(sp, text: str, mask_value: int):
    if not text:
        return [], []

    ids = sp.encode(text, out_type=int)
    mask = [mask_value] * len(ids)
    return ids, mask


def tokenize_sft_text(sp, text: str):
    """
    Builds token IDs and loss mask from formatted SmolTalk text.

    Mask rule:
      <|system|> tag/content      -> 0
      <|user|> tag/content        -> 0
      <|assistant|> tag           -> 0
      assistant content           -> 1
      <|endoftext|> after convo   -> 1

    The EOS token is included in loss because we want the model to learn
    when to stop the assistant response/conversation.
    """
    pieces = TAG_PATTERN.split(text)

    all_ids = []
    all_masks = []

    current_role = None

    for piece in tqdm(pieces, desc="Parsing/tagging text pieces"):
        if piece == "":
            continue

        if piece == SYSTEM_TAG:
            ids = sp.encode(piece, out_type=int)
            all_ids.extend(ids)
            all_masks.extend([0] * len(ids))
            current_role = "system"

        elif piece == USER_TAG:
            ids = sp.encode(piece, out_type=int)
            all_ids.extend(ids)
            all_masks.extend([0] * len(ids))
            current_role = "user"

        elif piece == ASSISTANT_TAG:
            ids = sp.encode(piece, out_type=int)
            all_ids.extend(ids)
            all_masks.extend([0] * len(ids))
            current_role = "assistant"

        elif piece == EOS_TAG:
            ids = sp.encode(piece, out_type=int)
            all_ids.extend(ids)

            # Usually EOS comes after assistant content, so include it in loss.
            # If weird formatting puts EOS elsewhere, don't train on it.
            eos_mask = 1 if current_role == "assistant" else 0
            all_masks.extend([eos_mask] * len(ids))

            current_role = None

        else:
            mask_value = 1 if current_role == "assistant" else 0
            ids, masks = encode_segment(sp, piece, mask_value)

            all_ids.extend(ids)
            all_masks.extend(masks)

    if len(all_ids) != len(all_masks):
        raise RuntimeError("Token IDs and masks have different lengths.")

    return all_ids, all_masks


def zero_contextless_leading_assistant_loss(chunks, masks, assistant_id, user_id, system_id):
    """
    If a chunk begins in the middle of an assistant answer, the original mask
    may start with 1s even though the user prompt is not inside the context.
    This zeros mask positions until the first explicit <|assistant|> tag
    appears in that chunk.

    This is conservative and usually improves SFT quality.
    """
    role_tag_ids = {assistant_id, user_id, system_id}

    for i in tqdm(range(len(chunks)), desc="Fixing leading contextless masks"):
        chunk = chunks[i]
        mask = masks[i]

        if mask[0] == 0:
            continue

        # If loss starts immediately, we may be in the middle of an assistant answer.
        # Zero loss until the next explicit role tag appears.
        found_role_tag = False

        for j, token_id in enumerate(chunk):
            if int(token_id) in role_tag_ids:
                found_role_tag = True

                # If the first role tag is assistant, keep loss after that tag.
                # If it is user/system, original mask should remain 0 until assistant later.
                mask[:j + 1] = 0
                break

        if not found_role_tag:
            # Whole chunk is probably a continuation of assistant content.
            # Safer to not train on it because there is no prompt in context.
            mask[:] = 0

    return masks


def save_chunks(token_ids, masks, out_tokens_path, out_masks_path, seq_len, token_dtype, fix_leading_masks, special_ids):
    chunk_len = seq_len + 1

    usable_len = (len(token_ids) // chunk_len) * chunk_len
    dropped = len(token_ids) - usable_len

    if usable_len == 0:
        raise ValueError(
            f"Not enough tokens to create one chunk of length {chunk_len}."
        )

    token_ids = np.asarray(token_ids[:usable_len], dtype=token_dtype)
    masks = np.asarray(masks[:usable_len], dtype=np.uint8)

    token_chunks = token_ids.reshape(-1, chunk_len)
    mask_chunks = masks.reshape(-1, chunk_len)

    if fix_leading_masks:
        mask_chunks = zero_contextless_leading_assistant_loss(
            chunks=token_chunks,
            masks=mask_chunks,
            assistant_id=special_ids["assistant"],
            user_id=special_ids["user"],
            system_id=special_ids["system"],
        )

    out_tokens_path = Path(out_tokens_path)
    out_masks_path = Path(out_masks_path)

    out_tokens_path.parent.mkdir(parents=True, exist_ok=True)
    out_masks_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(out_tokens_path, token_chunks)
    np.save(out_masks_path, mask_chunks)

    return {
        "tokens_path": str(out_tokens_path),
        "masks_path": str(out_masks_path),
        "num_tokens_total": len(token_ids) + dropped,
        "num_tokens_used": usable_len,
        "num_tokens_dropped": dropped,
        "num_sequences": token_chunks.shape[0],
        "chunk_len": chunk_len,
        "tokens_shape": token_chunks.shape,
        "masks_shape": mask_chunks.shape,
        "token_dtype": str(token_chunks.dtype),
        "mask_dtype": str(mask_chunks.dtype),
        "mask_ones": int(mask_chunks.sum()),
        "mask_fraction": float(mask_chunks.sum() / mask_chunks.size),
    }


def process_file(sp, txt_path, out_tokens_path, out_masks_path, seq_len, token_dtype, fix_leading_masks, special_ids):
    txt_path = Path(txt_path)

    print(f"\nReading {txt_path}...")
    text = txt_path.read_text(encoding="utf-8")

    print("Tokenizing and building loss mask...")
    token_ids, masks = tokenize_sft_text(sp, text)

    print(f"Total tokens: {len(token_ids):,}")
    print(f"Loss tokens:  {sum(masks):,}")
    print(f"Loss fraction before chunk fix: {sum(masks) / max(len(masks), 1):.4f}")

    stats = save_chunks(
        token_ids=token_ids,
        masks=masks,
        out_tokens_path=out_tokens_path,
        out_masks_path=out_masks_path,
        seq_len=seq_len,
        token_dtype=token_dtype,
        fix_leading_masks=fix_leading_masks,
        special_ids=special_ids,
    )

    return stats


def print_stats(name, stats):
    print(f"\n{name}:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--train_txt", type=str, default=DEFAULT_TRAIN_TXT)
    parser.add_argument("--val_txt", type=str, default=DEFAULT_VAL_TXT)
    parser.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR)

    parser.add_argument("--seq_len", type=int, default=1024)

    # Recommended: keep enabled.
    parser.add_argument(
        "--fix_leading_masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zero assistant loss at the start of chunks if the chunk starts mid-answer.",
    )

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
    verify_special_tokens_are_single(sp)
    special_ids = get_special_ids(sp)
    token_dtype = choose_token_dtype(sp)

    print("Tokenizer loaded.")
    print(f"Vocab size: {sp.vocab_size():,}")
    print(f"Using token dtype: {token_dtype}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Saved chunk length: {args.seq_len + 1}")
    print(f"Special IDs: {special_ids}")
    print(f"Fix leading contextless masks: {args.fix_leading_masks}")

    train_tokens_out = out_dir / f"smoltalk_train_seq{args.seq_len + 1}.npy"
    train_masks_out = out_dir / f"smoltalk_train_mask_seq{args.seq_len + 1}.npy"

    val_tokens_out = out_dir / f"smoltalk_val_seq{args.seq_len + 1}.npy"
    val_masks_out = out_dir / f"smoltalk_val_mask_seq{args.seq_len + 1}.npy"

    train_stats = process_file(
        sp=sp,
        txt_path=train_txt,
        out_tokens_path=train_tokens_out,
        out_masks_path=train_masks_out,
        seq_len=args.seq_len,
        token_dtype=token_dtype,
        fix_leading_masks=args.fix_leading_masks,
        special_ids=special_ids,
    )

    val_stats = process_file(
        sp=sp,
        txt_path=val_txt,
        out_tokens_path=val_tokens_out,
        out_masks_path=val_masks_out,
        seq_len=args.seq_len,
        token_dtype=token_dtype,
        fix_leading_masks=args.fix_leading_masks,
        special_ids=special_ids,
    )

    print("\nDone.")
    print_stats("Train", train_stats)
    print_stats("Val", val_stats)

    print("\nTraining usage later:")
    print(f"  tokens = np.load('{train_tokens_out}', mmap_mode='r')")
    print(f"  masks  = np.load('{train_masks_out}', mmap_mode='r')")
    print("  batch_tokens = tokens[idx]")
    print("  batch_masks  = masks[idx]")
    print("  x = batch_tokens[:, :-1]")
    print("  y = batch_tokens[:, 1:]")
    print("  loss_mask = batch_masks[:, 1:]")


if __name__ == "__main__":
    main()