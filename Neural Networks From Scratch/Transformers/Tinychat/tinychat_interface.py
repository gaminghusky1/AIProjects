import time

import sentencepiece as spm
from mlx_model import *
import mlx.core as mx
import numpy as np

def sample_next_token(probs, temperature=1.0):
    # probs[slash_inst_id] = 0.0
    probs = probs / mx.sum(probs)

    if temperature <= 0:
        return int(mx.argmax(probs).item())

    logits = mx.log(probs + 1e-9) / temperature
    logits_exp = mx.exp(logits)
    probs = logits_exp / mx.sum(logits_exp)

    probs_np = np.array(probs)
    return int(np.random.choice(len(probs_np), p=probs_np))

def generate_stream(transformer_model, prompt_ids, max_new_tokens=200, temperature=1.0, max_context=256):
    prompt_ids = prompt_ids[-max_context:]
    ids = mx.array([prompt_ids], dtype=mx.int32)

    generated_ids = []
    running_text = ""

    transformer_model.reset_cache()

    probs = transformer_model.predict(ids)
    next_probs = probs[0, -1, :]

    for _ in range(max_new_tokens):
        next_id = sample_next_token(next_probs, temperature)

        mx.eval(next_probs)
        mx.clear_cache()

        if next_id == inst_id or next_id == slash_inst_id:
            break

        generated_ids.append(next_id)

        # Keep full context ids for returning/truncating.
        next_token_array = mx.array([[next_id]], dtype=mx.int32)
        ids = mx.concatenate([ids, next_token_array], axis=1)

        # If context grows beyond max_context, reset cache and prefill again.
        if ids.shape[1] > max_context:
            ids = ids[:, -max_context:]

            transformer_model.reset_cache()

            probs = transformer_model.predict(ids)
            next_probs = probs[0, -1, :]

        else:
            # Normal cached generation:
            # feed ONLY the new token, not the whole growing ids.
            probs = transformer_model.predict(next_token_array)
            next_probs = probs[0, -1, :]

        full_text = sp.DecodeIds(generated_ids)
        new_text = full_text[len(running_text):]
        running_text = full_text

        yield {
            "type": "token",
            "id": next_id,
            "text": new_text
        }

        if next_id == sp.eos_id():
            break

    # assistant_output = sp.DecodeIds(generated_ids)
    new_ids = ids[0].tolist()

    yield {
        "type": "end",
        "generated_ids": generated_ids,
        "new_ids": new_ids
    }

    # return assistant_output, new_ids, generated_ids

def main():
    transformer_model = model.Model.load_from("TinychatModels/tinychat_model_batch_15000")

    # data = np.load("tinychat_ids.npy", mmap_mode="r")
    # print(sp.IdToPiece([int(a) for a in data[:1000]]))

    accumulated_ids = []
    max_context = 256

    # accumulated_ids.extend(sp.EncodeAsIds("[INST] Do you like listening to music? "))

    while True:
        raw_input = input("> ").strip()
        if raw_input == "C":
            accumulated_ids = []
            transformer_model.reset_cache()
            print("Context Cleared.")
            continue
        user_input = "[INST] " + raw_input + " [/INST] "
        user_input_ids = sp.EncodeAsIds(user_input)
        accumulated_ids.extend(user_input_ids)

        if len(accumulated_ids) > max_context:
            accumulated_ids = accumulated_ids[-max_context:]

        print(f"Context Length: {len(accumulated_ids)}")
        print(f"User Tokens: {sp.EncodeAsPieces(user_input)}")

        for event in generate_stream(transformer_model, accumulated_ids, max_new_tokens=100, temperature=0.7, max_context=max_context):
            if event["type"] == "token":
                print(event["text"], end="", flush=True)
            elif event["type"] == "end":
                accumulated_ids = event["new_ids"]
                print()

        # assistant_output, accumulated_ids, generated_ids = generate_stream(transformer_model, accumulated_ids, max_new_tokens=100, temperature=0.7)
        # print(sp.IdToPiece(accumulated_ids))
        # print("Assistant Tokens:", sp.IdToPiece(generated_ids))
        # print(assistant_output)

if __name__ == "__main__":
    sp = spm.SentencePieceProcessor()
    sp.Load("tinychat_tokenizer/spm.model")
    inst_id = sp.PieceToId("[INST]")
    slash_inst_id = sp.PieceToId("[/INST]")
    main()