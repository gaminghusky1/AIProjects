import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("spm_small.model")

text = "I love robotics and transformers!"

ids = sp.Encode(text, out_type=int)
pieces = sp.Encode(text, out_type=str)

print("IDs:", ids)
print("Pieces:", pieces)

decoded = sp.Decode(ids)
print("Decoded:", decoded)

print("Vocab size:", sp.GetPieceSize())
print("unk_id/bos_id/eos_id/pad_id:", sp.unk_id(), sp.bos_id(), sp.eos_id(), sp.pad_id())
