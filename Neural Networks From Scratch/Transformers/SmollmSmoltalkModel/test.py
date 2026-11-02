import numpy as np
import sentencepiece as spm

data = np.load("Data/tokenized/smoltalk_train_seq1025.npy")
mask = np.load("Data/tokenized/smoltalk_train_mask_seq1025.npy")

sp = spm.SentencePieceProcessor()
sp.Load("Tokenizer/smollm_smoltalk_bpe_32k.model")

for i in range(data[0].shape[0]):
    if not mask[0][i]:
        print(sp.Decode(int(data[0, i])), end="")