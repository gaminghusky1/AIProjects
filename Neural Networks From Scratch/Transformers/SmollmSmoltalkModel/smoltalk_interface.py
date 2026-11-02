import sentencepiece as spm
from mlx_model import *
import mlx.core as mx
import numpy as np

def apply_repetition_penalty(logits, generated_ids, penalty=1.0):
    if penalty == 1.0 or len(generated_ids) == 0:
        return logits

    logits_np = np.array(logits)

    for token_id in set(generated_ids):
        if logits_np[token_id] > 0:
            logits_np[token_id] /= penalty
        else:
            logits_np[token_id] *= penalty

    return mx.array(logits_np)

def sample_next_token(logits, temperature=1.0, top_p=1.0, top_k=None, generated_ids=None, repetition_penalty=1.0):
    if generated_ids is not None:
        logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    if temperature == 0:
        return int(mx.argmax(logits))

    logits = logits / temperature

    if top_k is not None and 0 < top_k <= len(logits):
        keep_indices = mx.argpartition(logits, -top_k)[-top_k:]
        mask = mx.full(logits.shape, False, dtype=mx.bool_)
        mask[keep_indices] = True
        logits = mx.where(mask, logits, -mx.inf)

    probs = mx.softmax(logits)
    probs_np = np.array(probs)

    if top_p < 1.0:
        sorted_idx = np.argsort(probs_np)[::-1]
        sorted_probs = probs_np[sorted_idx]

        prefix_summed_probs = np.cumsum(sorted_probs)

        keep = prefix_summed_probs <= top_p
        keep[0] = True

        filtered_idx = sorted_idx[keep]
        filtered_probs = sorted_probs[keep]

        filtered_probs /= np.sum(filtered_probs)

        return int(np.random.choice(filtered_idx, p=filtered_probs))

    return int(np.random.choice(len(probs_np), p=probs_np))

def generate_stream(transformer_model, prompt_ids, max_new_tokens=200, temperature=1.0, top_p=1.0, top_k=None, repetition_penalty=1.0, max_context=256):
    # truncate prompt before creating tensor
    prompt_ids = prompt_ids[-max_context:]
    ids = mx.array([prompt_ids], dtype=mx.int32)

    generated_ids = []
    running_text = ""
    rebuild_context = int(max_context * 0.75)

    transformer_model.reset_cache()

    logits = transformer_model.predict(ids)
    next_logits = logits[0, -1, :]

    for _ in range(max_new_tokens):
        next_id = sample_next_token(
            next_logits,
            temperature,
            top_p,
            top_k,
            generated_ids,
            repetition_penalty,
        )

        mx.eval(next_logits)
        mx.clear_cache()

        if next_id == end_id or next_id == user_id:
            break

        generated_ids.append(next_id)

        # Keep full context ids for returning/truncating.
        next_token_array = mx.array([[next_id]], dtype=mx.int32)
        ids = mx.concatenate([ids, next_token_array], axis=1)

        # If context grows beyond max_context, reset cache and prefill again.
        # This is RoPE-safe because positions restart consistently for the truncated context.
        if ids.shape[1] > max_context:
            ids = ids[:, -rebuild_context:]

            transformer_model.reset_cache()

            logits = transformer_model.predict(ids)
            next_logits = logits[0, -1, :]

        else:
            # Normal cached generation:
            # feed ONLY the new token, not the whole growing ids.
            logits = transformer_model.predict(next_token_array)
            next_logits = logits[0, -1, :]

        full_text = sp.DecodeIds(generated_ids)
        new_text = full_text[len(running_text):]
        running_text = full_text

        yield {
            "type": "token",
            "id": next_id,
            "text": new_text,
        }

    yield {
        "type": "end",
        "generated_ids": generated_ids,
        "new_ids": ids[0].tolist(),
    }


def main():
    transformer_model = model.Model.load_from("Models/smoltalk_v1_e1_final")

    # data = np.load("tinychat_ids.npy", mmap_mode="r")
    # print(sp.IdToPiece([int(a) for a in data[:1000]]))

    accumulated_ids = []
    max_context = 1024

    # accumulated_ids.extend(sp.EncodeAsIds("[INST] Do you like listening to music? "))

    while True:
        raw_input = input("> ").strip()
        if raw_input == "C":
            accumulated_ids = []
            transformer_model.reset_cache()
            print("Context Cleared.")
            continue
        user_input =  f"<|user|>\n{raw_input}\n\n<|assistant|>\n"
        user_input_ids = sp.EncodeAsIds(user_input)
        accumulated_ids.extend(user_input_ids)

        if len(accumulated_ids) > max_context:
            accumulated_ids = accumulated_ids[-max_context:]

        print(f"Context Length: {len(accumulated_ids)}")
        print(f"User Tokens: {sp.EncodeAsPieces(user_input)}")

        for event in generate_stream(transformer_model, accumulated_ids, max_new_tokens=2000, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.2, max_context=max_context):
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
    sp.Load("Tokenizer/smollm_smoltalk_bpe_32k.model")
    end_id = sp.PieceToId("<|endoftext|>")
    user_id = sp.PieceToId("<|user|>")
    main()