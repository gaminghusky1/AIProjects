import sentencepiece as spm
from mlx_model import *
import mlx.core as mx
import numpy as np

def sample_next_token(logits, temperature=1.0):
    if temperature <= 0:
        return int(mx.argmax(logits).item())

    logits = logits / temperature
    probs = mx.softmax(logits)

    probs_np = np.array(probs)
    return int(np.random.choice(len(probs_np), p=probs_np))

def generate(transformer_model, sp, prompt_ids, max_new_tokens=200, temperature=1.0):
    ids = mx.array([prompt_ids], dtype=mx.int32)

    generated_ids = []

    for _ in range(max_new_tokens):
        logits = transformer_model.predict(ids)
        next_logits = logits[0, -1, :]
        next_id = sample_next_token(next_logits, temperature)

        generated_ids.append(next_id)

        ids = mx.concatenate([ids, mx.array([[next_id]], dtype=mx.int32)], axis=1)

        if next_id == sp.eos_id():
            break

    assistant_output = sp.DecodeIds(generated_ids)
    new_ids = ids[0].tolist()

    return assistant_output.rstrip(), new_ids

def main():
    sp = spm.SentencePieceProcessor()
    sp.Load("tinychat_tokenizer/spm.model")

    transformer_model = model.Model.load_from("TinychatModels/tinychat_model_batch_3000")

    accumulated_ids = []
    max_context = 128

    while True:
        user_input = "[INST] " + input("> ").strip() + " [/INST]"
        user_input_ids = sp.EncodeAsIds(user_input)
        accumulated_ids.extend(user_input_ids)

        assistant_output, accumulated_ids = generate(transformer_model, sp, accumulated_ids, max_new_tokens=200, temperature=0.9)
        print(assistant_output)

        if len(accumulated_ids) > max_context:
            accumulated_ids = accumulated_ids[-max_context:]

if __name__ == "__main__":
    main()