import mlx.core as mx
from datasets import load_dataset
import sentencepiece as spm
import numpy as np

DATASET_NAME = "starhopp3r/TinyChat"

def stream_tinychat_text(split="train"):
    ds = load_dataset(DATASET_NAME, split=split, streaming=True)
    for ex in ds:
        txt = ex.get("text", "")
        if txt:
            yield txt

def main():
    sp = spm.SentencePieceProcessor()
    sp.Load("tinychat_tokenizer/spm.model")

    eos = sp.eos_id()

    ids = []
    i = 0
    for txt in stream_tinychat_text():
        ids.extend(sp.EncodeAsIds(txt))
        if eos != -1:
            ids.append(eos)
        if i >= 2:
            break
        i += 1
    print(len(ids))
    print(sp.IdToPiece(ids))
    data = np.load("old_tinychat_tokenizer/tinychat_tokens.npy", mmap_mode="r")
    print(sp.IdToPiece([int(a) for a in data[:len(ids)]]))

if __name__ == "__main__":
    main()