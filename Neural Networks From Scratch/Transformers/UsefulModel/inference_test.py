import sentencepiece as spm
from mlx_model import *
import mlx.core as mx
import numpy as np

def sample_next_token(logits, temperature=1.0):
    if temperature == 0:
        return int(mx.argmax(logits))

    probs = mx.softmax(logits / temperature)
    probs_np = np.array(probs)

    return int(np.random.choice(len(probs_np), p=probs_np))

def generate_stream(transformer_model, prompt_ids, max_new_tokens=200, temperature=1.0, max_context=256):
    # truncate prompt before creating tensor
    prompt_ids = prompt_ids[-max_context:]
    ids = mx.array([prompt_ids], dtype=mx.int32)

    generated_ids = []
    running_text = ""

    for _ in range(max_new_tokens):
        logits = transformer_model.predict(ids)
        next_logits = logits[0, -1, :]
        next_id = sample_next_token(next_logits, temperature)

        if next_id == end_id:
            print("\n ENDED \n")
            break

        ids = mx.concatenate([ids, mx.array([[next_id]], dtype=mx.int32)], axis=1)

        # keep only the last max_context tokens
        if ids.shape[1] > max_context:
            ids = ids[:, -max_context:]

        generated_ids.append(next_id)

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

    yield {
        "type": "end",
        "generated_ids": generated_ids,
        "new_ids": ids[0].tolist()
    }

def main():
    transformer_model = model.Model.load_from("Models/useful_model_batch_20000")

    # data = np.load("tinychat_ids.npy", mmap_mode="r")
    # print(sp.IdToPiece([int(a) for a in data[:1000]]))

    accumulated_ids = []
    max_context = 256

    # accumulated_ids.extend(sp.EncodeAsIds("[INST] Do you like listening to music? "))

    while True:
        raw_input = input("> ").strip()
        if raw_input == "C":
            accumulated_ids = []
            print("Context Cleared.")
            continue
        user_input = "<|document|> " + raw_input + " "
        user_input_ids = sp.EncodeAsIds(user_input)
        accumulated_ids.extend(user_input_ids)

        if len(accumulated_ids) > max_context:
            accumulated_ids = accumulated_ids[-max_context:]

        print(f"Context Length: {len(accumulated_ids)}")
        print(f"User Tokens: {sp.EncodeAsPieces(user_input)}")

        for event in generate_stream(transformer_model, accumulated_ids, max_new_tokens=1000, temperature=0.7, max_context=max_context):
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
    sp.Load("Tokenizer/useful_spm.model")
    start_id = sp.PieceToId("<|document|>")
    end_id = sp.PieceToId("<|endoftext|>")
    main()