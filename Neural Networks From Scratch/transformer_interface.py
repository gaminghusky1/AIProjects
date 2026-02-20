import sentencepiece as spm
import layers
import model
import numpy as np

def main():
    sp = spm.SentencePieceProcessor()
    sp.Load("TransformerData/spm_dialogue.model")

    transformer_model = model.Model.load_from("Models/transformer_test")

    previous_tokens = []

    while True:
        user_input = ">persona< " + input("> ") + " >endoftext<\n>personb<"
        previous_tokens.extend(sp.EncodeAsIds(user_input))  # EXTEND, not append

        end_id = sp.PieceToId(">endoftext<")  # better than EncodeAsIds([..])

        model_output_text = ""
        model_output_token = None

        while model_output_token != end_id:
            x = np.array(previous_tokens, dtype=np.int64)[None, :]  # (1, T)
            prediction = transformer_model.predict(x)  # should be (1, T, V) or (1, V)

            # If prediction is (1, T, V), take last timestep:
            logits_last = prediction[0, -1, :]
            model_output_token = int(np.argmax(logits_last))

            model_output_text += sp.DecodeIds([model_output_token])
            previous_tokens.append(model_output_token)
            print(model_output_text, "\n")

        previous_tokens.extend(sp.EncodeAsIds("\n"))
        print(model_output_text.rstrip(" "))

if __name__ == "__main__":
    main()