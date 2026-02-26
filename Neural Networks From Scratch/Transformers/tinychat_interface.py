import sentencepiece as spm
from mlx_model import *
import mlx.core as mx
import numpy as np

def sample_next_token(probs, temperature=1.0):
    probs[slash_inst_id] = 0.0
    probs = probs / mx.sum(probs)

    if temperature <= 0:
        return int(mx.argmax(probs).item())

    logits = mx.log(probs + 1e-9) / temperature
    logits_exp = mx.exp(logits)
    probs = logits_exp / mx.sum(logits_exp)

    probs_np = np.array(probs)
    return int(np.random.choice(len(probs_np), p=probs_np))

def generate(transformer_model, prompt_ids, max_new_tokens=200, temperature=1.0):
    ids = mx.array([prompt_ids], dtype=mx.int32)

    generated_ids = []

    for _ in range(max_new_tokens):
        probs = transformer_model.predict(ids)
        next_probs = probs[0, -1, :]
        next_id = sample_next_token(next_probs, temperature)

        if next_id == inst_id:
            break

        generated_ids.append(next_id)

        ids = mx.concatenate([ids, mx.array([[next_id]], dtype=mx.int32)], axis=1)

        if next_id == sp.eos_id():
            break

    # text1 = sp.DecodeIds(generated_ids)
    #
    # pieces = [sp.IdToPiece(t) for t in generated_ids]
    # text2 = sp.DecodePieces(pieces)
    #
    # print("ids decode:", repr(text1[:200]))
    # print("pieces decode:", repr(text2[:200]))
    # print("pieces near end:", [repr(p) for p in pieces[-30:]])
    assistant_output = sp.DecodeIds(generated_ids)
    new_ids = ids[0].tolist()

    return assistant_output, new_ids, generated_ids

def main():
    transformer_model = model.Model.load_from("TinychatModels/tinychat_model_batch_1000")

    accumulated_ids = []
    max_context = 256

    while True:
        user_input = "[INST] " + input("> ").strip() + " [/INST] "
        user_input_ids = sp.EncodeAsIds(user_input)
        accumulated_ids.extend(user_input_ids)

        assistant_output, accumulated_ids, generated_ids = generate(transformer_model, accumulated_ids, max_new_tokens=50, temperature=0.7)
        print(sp.EncodeAsPieces(user_input))
        print(sp.IdToPiece(generated_ids))
        print(assistant_output)

        if len(accumulated_ids) > max_context:
            accumulated_ids = accumulated_ids[-max_context:]

if __name__ == "__main__":
    sp = spm.SentencePieceProcessor()
    sp.Load("tinychat_tokenizer/spm.model")
    inst_id = sp.PieceToId("[INST]")
    slash_inst_id = sp.PieceToId("[/INST]")
    main()