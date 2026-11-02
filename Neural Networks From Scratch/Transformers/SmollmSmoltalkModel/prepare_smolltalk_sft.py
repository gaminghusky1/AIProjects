# prepare_smoltalk_sft_txt.py
#
# Install:
#   pip install datasets tqdm datasketch
#
# Run:
#   python prepare_smoltalk_sft_txt.py
#
# Output:
#   Data/smoltalk_train.txt
#   Data/smoltalk_val.txt

import argparse
import hashlib
import html
import random
import re
import unicodedata
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False


DATASET_NAME = "HuggingFaceTB/smoltalk"
DATASET_CONFIG = "all"

SYSTEM_TAG = "<|system|>"
USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"
EOS_TAG = "<|endoftext|>"

SPECIAL_TAGS = [
    SYSTEM_TAG,
    USER_TAG,
    ASSISTANT_TAG,
    EOS_TAG,
]


def utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def remove_bad_unicode_chars(text: str) -> str:
    cleaned = []

    for ch in text:
        code = ord(ch)
        category = unicodedata.category(ch)

        # Keep useful whitespace.
        if ch in ("\n", "\t"):
            cleaned.append(ch)
            continue

        # Remove ASCII control chars except newline/tab.
        if code < 32 or code == 127:
            continue

        # Remove invisible/control Unicode junk:
        # Cc = control, Cf = format, Cs = surrogate.
        if category in ("Cc", "Cf", "Cs"):
            continue

        cleaned.append(ch)

    return "".join(cleaned)


def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)

    # Do not let actual training text contain your chat control tokens.
    for tag in SPECIAL_TAGS:
        text = text.replace(tag, "")

    text = text.replace("\uFFFD", "")
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = remove_bad_unicode_chars(text)

    # For SFT, tabs are usually not meaningful unless inside code.
    # Keep code indentation mostly intact by only replacing long tab runs.
    text = text.replace("\t", "    ")

    # Normalize excessive spaces/newlines.
    text = re.sub(r"[ \t]{5,}", "    ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text.strip()


def normalize_role(role: str) -> str | None:
    if role is None:
        return None

    role = str(role).lower().strip()

    role_map = {
        "system": "system",
        "user": "user",
        "human": "user",
        "prompter": "user",
        "instruction": "user",
        "assistant": "assistant",
        "gpt": "assistant",
        "bot": "assistant",
        "model": "assistant",
        "completion": "assistant",
    }

    return role_map.get(role)


def extract_messages(example):
    """
    SmolTalk usually uses a conversational field such as "messages".
    This also handles common alternative names.
    """
    possible_keys = [
        "messages",
        "conversations",
        "conversation",
        "dialogue",
        "dialog",
    ]

    raw_messages = None
    for key in possible_keys:
        if key in example and example[key]:
            raw_messages = example[key]
            break

    if raw_messages is None:
        return None

    if not isinstance(raw_messages, list):
        return None

    messages = []

    for msg in raw_messages:
        if not isinstance(msg, dict):
            return None

        role = (
            msg.get("role")
            or msg.get("from")
            or msg.get("speaker")
            or msg.get("author")
        )

        content = (
            msg.get("content")
            or msg.get("value")
            or msg.get("text")
            or msg.get("message")
        )

        role = normalize_role(role)
        content = clean_text(content)

        if role is None:
            return None

        # For a simple chatbot, skip function/tool-call data.
        if role not in ("system", "user", "assistant"):
            return None

        if not content:
            return None

        messages.append({"role": role, "content": content})

    return messages


def merge_consecutive_same_roles(messages):
    if not messages:
        return []

    merged = []

    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged.append(dict(msg))

    return merged


def valid_conversation(messages, min_chars=100, max_chars=12_000):
    if not messages:
        return False

    messages = merge_consecutive_same_roles(messages)

    # Allow system only at the beginning.
    non_system_seen = False
    for msg in messages:
        if msg["role"] != "system":
            non_system_seen = True
        elif non_system_seen:
            return False

    non_system = [m for m in messages if m["role"] != "system"]

    if len(non_system) < 2:
        return False

    if non_system[0]["role"] != "user":
        return False

    if not any(m["role"] == "assistant" for m in non_system):
        return False

    # Require user/assistant alternation after optional system.
    expected = "user"
    for msg in non_system:
        if msg["role"] != expected:
            return False
        expected = "assistant" if expected == "user" else "user"

    # Prefer conversations ending with assistant response.
    if non_system[-1]["role"] != "assistant":
        return False

    total_chars = sum(len(m["content"]) for m in messages)
    if total_chars < min_chars:
        return False

    if total_chars > max_chars:
        return False

    # Filter obvious repeated-line junk.
    all_text = "\n".join(m["content"] for m in messages)
    lines = [line.strip() for line in all_text.splitlines() if line.strip()]
    if len(lines) >= 12:
        unique_ratio = len(set(lines)) / len(lines)
        if unique_ratio < 0.55:
            return False

    # Filter excessive URL/html residue.
    lower = all_text.lower()
    if lower.count("http://") + lower.count("https://") > 15:
        return False

    if lower.count("<div") + lower.count("</div>") + lower.count("<span") > 4:
        return False

    # Filter very repetitive character spam.
    if re.search(r"(.)\1{40,}", all_text):
        return False

    return True


def format_conversation(messages):
    messages = merge_consecutive_same_roles(messages)

    blocks = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            blocks.append(f"{SYSTEM_TAG}\n{content}")
        elif role == "user":
            blocks.append(f"{USER_TAG}\n{content}")
        elif role == "assistant":
            blocks.append(f"{ASSISTANT_TAG}\n{content}")
        else:
            raise ValueError(f"Unexpected role: {role}")

    # Only one EOS at the end of the entire conversation.
    return "\n\n".join(blocks).strip() + f"\n{EOS_TAG}"


def normalized_for_hash(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def exact_hash(text: str) -> str:
    return hashlib.sha1(normalized_for_hash(text).encode("utf-8")).hexdigest()


def assistant_answer_hashes(messages):
    hashes = []
    for msg in messages:
        if msg["role"] == "assistant":
            content = normalized_for_hash(msg["content"])
            if len(content) > 50:
                hashes.append(hashlib.sha1(content.encode("utf-8")).hexdigest())
    return hashes


def shingles(text: str, n: int = 5):
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return []
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def minhash_text(text: str, num_perm: int = 64):
    mh = MinHash(num_perm=num_perm)
    for shingle in shingles(text):
        mh.update(shingle.encode("utf-8"))
    return mh


def split_by_val_bytes(conversations, val_frac):
    total_bytes = sum(utf8_len(c) + 2 for c in conversations)
    target_val_bytes = int(total_bytes * val_frac)

    val = []
    train = []
    val_bytes = 0

    for conv in conversations:
        if val_bytes < target_val_bytes:
            val.append(conv)
            val_bytes += utf8_len(conv) + 2
        else:
            train.append(conv)

    return train, val


def write_txt(path: Path, conversations):
    with path.open("w", encoding="utf-8") as f:
        for conv in conversations:
            f.write(conv)
            f.write("\n\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", type=str, default="Data")
    parser.add_argument("--train_name", type=str, default="smoltalk_train.txt")
    parser.add_argument("--val_name", type=str, default="smoltalk_val.txt")

    # Your SmolLM pretrain file was ~433 MB train.
    # SFT should usually be much smaller. 60 MB total is about 14% of that,
    # which is a reasonable first SFT size for a small model.
    parser.add_argument("--target_total_mb", type=float, default=60.0)

    parser.add_argument("--val_frac", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--min_chars", type=int, default=100)
    parser.add_argument("--max_chars", type=int, default=12_000)

    # Safety cap. Usually target_total_mb stops first.
    parser.add_argument("--max_conversations", type=int, default=100_000)

    # Near dedup is enabled by default.
    parser.add_argument("--near_dedup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--near_dedup_threshold", type=float, default=0.86)

    # Streaming shuffle buffer.
    parser.add_argument("--shuffle_buffer", type=int, default=20_000)

    args = parser.parse_args()
    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_bytes = int(args.target_total_mb * 1024 * 1024)

    print("Loading SmolTalk...")
    ds = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        split="train",
        streaming=True,
    )

    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    seen_conversation_hashes = set()
    assistant_hash_counts = {}

    if args.near_dedup and HAS_DATASKETCH:
        lsh = MinHashLSH(
            threshold=args.near_dedup_threshold,
            num_perm=64,
        )
    elif args.near_dedup and not HAS_DATASKETCH:
        print("WARNING: datasketch is not installed, so near dedup is disabled.")
        print("Install it with: pip install datasketch")
        lsh = None
    else:
        lsh = None

    conversations = []
    total_bytes = 0

    stats = {
        "seen": 0,
        "bad_format": 0,
        "filtered": 0,
        "exact_duplicate": 0,
        "near_duplicate": 0,
        "assistant_duplicate": 0,
        "kept": 0,
    }

    pbar = tqdm(total=target_bytes, desc="Collecting SFT text bytes")

    for example in ds:
        stats["seen"] += 1

        messages = extract_messages(example)
        if messages is None:
            stats["bad_format"] += 1
            continue

        messages = merge_consecutive_same_roles(messages)

        if not valid_conversation(
            messages,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        ):
            stats["filtered"] += 1
            continue

        formatted = format_conversation(messages)

        h = exact_hash(formatted)
        if h in seen_conversation_hashes:
            stats["exact_duplicate"] += 1
            continue
        seen_conversation_hashes.add(h)

        # Avoid training many copies of exactly the same assistant answer.
        duplicate_answer = False
        for ah in assistant_answer_hashes(messages):
            count = assistant_hash_counts.get(ah, 0)
            if count >= 2:
                duplicate_answer = True
                break

        if duplicate_answer:
            stats["assistant_duplicate"] += 1
            continue

        if lsh is not None:
            sh = shingles(formatted)
            if len(sh) >= 20:
                mh = minhash_text(formatted)
                matches = lsh.query(mh)
                if matches:
                    stats["near_duplicate"] += 1
                    continue
                lsh.insert(h, mh)

        for ah in assistant_answer_hashes(messages):
            assistant_hash_counts[ah] = assistant_hash_counts.get(ah, 0) + 1

        conversations.append(formatted)
        added = utf8_len(formatted) + 2
        total_bytes += added
        stats["kept"] += 1
        pbar.update(added)

        if total_bytes >= target_bytes:
            break

        if len(conversations) >= args.max_conversations:
            break

    pbar.close()

    print("\nCollection stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")

    print(f"\nCollected conversations: {len(conversations):,}")
    print(f"Collected size: {total_bytes / (1024 * 1024):.2f} MB")

    print("Final shuffle...")
    random.shuffle(conversations)

    train_convs, val_convs = split_by_val_bytes(conversations, args.val_frac)

    train_path = out_dir / args.train_name
    val_path = out_dir / args.val_name

    write_txt(train_path, train_convs)
    write_txt(val_path, val_convs)

    train_size = train_path.stat().st_size / (1024 * 1024)
    val_size = val_path.stat().st_size / (1024 * 1024)

    print("\nDone.")
    print(f"Train conversations: {len(train_convs):,}")
    print(f"Val conversations:   {len(val_convs):,}")
    print(f"Train path: {train_path} ({train_size:.2f} MB)")
    print(f"Val path:   {val_path} ({val_size:.2f} MB)")


if __name__ == "__main__":
    main()