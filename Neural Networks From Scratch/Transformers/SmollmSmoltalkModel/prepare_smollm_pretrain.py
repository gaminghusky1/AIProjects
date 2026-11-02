# prepare_smollm_pretrain_txt.py
# pip install datasets tqdm datasketch

import argparse
import hashlib
import html
import random
import re
import unicodedata
from pathlib import Path

from datasets import load_dataset, interleave_datasets
from tqdm import tqdm

try:
    from datasketch import MinHash, MinHashLSH
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False


DATASET_NAME = "HuggingFaceTB/smollm-corpus"

SUBSETS = {
    # Recommended proportions for a small general chatbot pretrain corpus.
    "fineweb-edu-dedup": 0.75,
    "cosmopedia-v2": 0.20,
    "python-edu": 0.05,
}

DOC_SEPARATOR = "\n<|endoftext|>\n"


def get_text(example):
    for key in ("text", "content", "code"):
        if key in example and example[key]:
            return example[key]
    return ""


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)

    # Remove broken/control characters.
    text = text.replace("\uFFFD", "")
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)

    # Normalize newlines/spaces.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]{3,}", "  ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    # Remove common web junk.
    junk_lines = [
        r"^skip to (main )?content$",
        r"^subscribe to our newsletter$",
        r"^all rights reserved\.?$",
        r"^cookie policy$",
        r"^privacy policy$",
        r"^terms of service$",
    ]

    cleaned_lines = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            cleaned_lines.append("")
            continue

        lower = s.lower()
        if any(re.match(p, lower) for p in junk_lines):
            continue

        cleaned_lines.append(s)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def looks_bad(text: str, is_code: bool = False) -> bool:
    if len(text) < (80 if is_code else 200):
        return True

    if len(text) > 100_000:
        return True

    # Too much URL/html residue.
    lower = text.lower()
    if lower.count("http://") + lower.count("https://") > 20:
        return True

    if lower.count("<div") + lower.count("</div>") + lower.count("<span") > 5:
        return True

    # Too many weird symbols relative to total length.
    weird = sum(1 for c in text if ord(c) > 127 and not c.isalpha() and not c.isspace())
    if weird / max(len(text), 1) > 0.03:
        return True

    # Repeated-line junk.
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= 10:
        unique_ratio = len(set(lines)) / len(lines)
        if unique_ratio < 0.55:
            return True

    # Extremely repetitive characters.
    if re.search(r"(.)\1{40,}", text):
        return True

    return False


def exact_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def shingles(text: str, n: int = 5):
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return []
    return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]


def minhash_text(text: str, num_perm: int = 64):
    mh = MinHash(num_perm=num_perm)
    for shingle in shingles(text):
        mh.update(shingle.encode("utf-8"))
    return mh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="smollm_pretrain_txt")
    parser.add_argument("--num_docs", type=int, default=100_000)
    parser.add_argument("--val_frac", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)

    # Near-dedup is useful, but slower. Exact dedup always runs.
    parser.add_argument("--near_dedup", action="store_true")
    parser.add_argument("--near_dedup_threshold", type=float, default=0.85)

    args = parser.parse_args()
    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading streaming datasets...")

    datasets = []
    probabilities = []

    for subset, prob in SUBSETS.items():
        ds = load_dataset(
            DATASET_NAME,
            subset,
            split="train",
            streaming=True,
        )

        # Shuffle each source buffer before interleaving.
        ds = ds.shuffle(seed=args.seed, buffer_size=20_000)

        datasets.append(ds)
        probabilities.append(prob)

    mixed = interleave_datasets(
        datasets,
        probabilities=probabilities,
        seed=args.seed,
        stopping_strategy="all_exhausted",
    )

    seen_exact = set()

    if args.near_dedup:
        if not HAS_DATASKETCH:
            raise RuntimeError(
                "Near dedup requires datasketch. Install with: pip install datasketch"
            )
        lsh = MinHashLSH(
            threshold=args.near_dedup_threshold,
            num_perm=64,
        )
    else:
        lsh = None

    docs = []
    source_counts = {k: 0 for k in SUBSETS}

    pbar = tqdm(total=args.num_docs, desc="Collecting cleaned docs")

    for example in mixed:
        raw = get_text(example)
        if not raw:
            continue

        text = clean_text(raw)

        # Guess whether this is from Python-Edu.
        # The interleaved examples do not directly expose the subset name,
        # so infer from code-like metadata/shape.
        is_code = "repo_name" in example or "path" in example or "def " in text[:1000]

        if looks_bad(text, is_code=is_code):
            continue

        h = exact_hash(text)
        if h in seen_exact:
            continue
        seen_exact.add(h)

        if args.near_dedup:
            sh = shingles(text)
            if len(sh) >= 20:
                mh = minhash_text(text)
                matches = lsh.query(mh)
                if matches:
                    continue
                lsh.insert(h, mh)

        docs.append(text)
        pbar.update(1)

        # Best-effort source count.
        if "repo_name" in example or "path" in example:
            source_counts["python-edu"] += 1
        elif "prompt" in example or "audience" in example:
            source_counts["cosmopedia-v2"] += 1
        else:
            source_counts["fineweb-edu-dedup"] += 1

        if len(docs) >= args.num_docs:
            break

    pbar.close()

    print(f"Collected {len(docs):,} documents.")
    print("Approx source counts:", source_counts)

    print("Shuffling final documents...")
    random.shuffle(docs)

    val_size = max(1, int(len(docs) * args.val_frac))
    val_docs = docs[:val_size]
    train_docs = docs[val_size:]

    train_path = out_dir / "train.txt"
    val_path = out_dir / "val.txt"

    print(f"Writing {train_path}...")
    with train_path.open("w", encoding="utf-8") as f:
        for doc in train_docs:
            f.write(doc)
            f.write(DOC_SEPARATOR)

    print(f"Writing {val_path}...")
    with val_path.open("w", encoding="utf-8") as f:
        for doc in val_docs:
            f.write(doc)
            f.write(DOC_SEPARATOR)

    print("Done.")
    print(f"Train docs: {len(train_docs):,}")
    print(f"Val docs:   {len(val_docs):,}")
    print(f"Output dir: {out_dir.resolve()}")


if __name__ == "__main__":
    main()