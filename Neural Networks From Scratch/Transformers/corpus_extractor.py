import os
from datasets import load_dataset, interleave_datasets
from tqdm import tqdm

OUT_DIR = "TransformerData"
os.makedirs(OUT_DIR, exist_ok=True)

MIX_PATH = os.path.join(OUT_DIR, "mixed_owt70_wt30.txt")

# How much text to extract (CPU-friendly starter):
# 200k docs is already big; start smaller if you want.
NUM_DOCS = 50_000

SEED = 42
SHUFFLE_BUFFER = 10_000  # streaming shuffle buffer (bigger = more random but slower)

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.strip()
    return s

def iter_docs():
    # OpenWebText: has a "text" field
    owt = load_dataset("Skylion007/openwebtext", split="train", streaming=True)  # :contentReference[oaicite:4]{index=4}
    # WikiText-103-v1: has a "text" field
    wt  = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split="train", streaming=True)  # :contentReference[oaicite:5]{index=5}

    # Shuffle within each stream a bit (approximate)
    owt = owt.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)
    wt  = wt.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER)

    # Interleave with probabilities 70/30
    mixed = interleave_datasets([owt, wt], probabilities=[0.7, 0.3], seed=SEED)

    # Yield cleaned docs
    for ex in mixed:
        txt = clean_text(ex.get("text", ""))
        # Filter junk/empty
        if len(txt) < 50:
            continue
        yield txt

def write_mixed_file(path: str, num_docs: int):
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for txt in tqdm(iter_docs(), total=num_docs, desc="Writing mixed corpus"):
            f.write("<doc>\n")
            f.write(txt.replace("\n", " "))  # 1 doc per line (simplifies SPM + packing)
            f.write("\n")
            n += 1
            if n >= num_docs:
                break
    print(f"Wrote {n} docs to {path}")

if __name__ == "__main__":
    write_mixed_file(MIX_PATH, NUM_DOCS)