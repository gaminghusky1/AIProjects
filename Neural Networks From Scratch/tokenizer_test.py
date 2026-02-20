import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("TransformerData/spm_dialogue.model")

print(sp.EncodeAsPieces(">persona< Hi there >endoftext<"))