import os
import sentencepiece as spm

OUT_DIR = "TransformerData"
MIX_PATH = os.path.join(OUT_DIR, "mixed_owt70_wt30.txt")

SPM_PREFIX = os.path.join(OUT_DIR, "spm_openwebtext_wikitext")
VOCAB_SIZE = 8000

# Special token IDs:
# We'll reserve:
# 0 = <pad>, 1 = <unk>, 2 = <s>, 3 = </s>
spm.SentencePieceTrainer.Train(
    input=MIX_PATH,
    model_prefix=SPM_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type="bpe",         # good default; unigram also works
    character_coverage=1.0,   # English
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
)
print("Trained:", SPM_PREFIX + ".model", SPM_PREFIX + ".vocab")