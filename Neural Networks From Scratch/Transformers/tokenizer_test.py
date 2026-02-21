import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("TransformerData/spm_dialogue.base_model")

print(sp.EncodeAsPieces(">persona< Hi there >endoftext<"))