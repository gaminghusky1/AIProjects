from base_model import *
import numpy as np
import sentencepiece as spm

def main():
    sentence = "Narbonne Cathedral is a Catholic church located in the town of Narbonne, France. Dedicated to Saints Justus and Pastor, it was the cathedral of the Diocese of Narbonne until it was merged with the Diocese of Carcassonne under the Concordat of 1801. It is now a co-cathedral of the Diocese of Carcassonne–Narbonne, and was declared a minor basilica in 1886. The first church on the site was a small Constantinian structure that was erected in 313 and destroyed by fire in 441. A replacement building, erected in 445, fell into ruin and was eventually replaced in 890 by a Carolingian cathedral whose restored steeple remains on the site. Construction on the present Gothic building began in 1272, opening in 1286. It was gradually expanded until 1354, but its size was then limited by the location of the city walls and the rest of the building was never completed, the nave and transept being notably absent. This photograph shows the choir of Narbonne cathedral, looking towards the high altar in the background."
    sp = spm.SentencePieceProcessor()
    sp.Load("TransformerData/spm_small.base_model")
    tokens = np.array(sp.Encode(sentence, out_type=int))
    L = 1
    if len(tokens) % L != 0:
        tokens = np.concatenate([tokens, np.zeros((L - len(tokens) % L), dtype=int)])
    tokens = tokens.reshape((-1, L))
    print(tokens)
    embedding_test_model = model.Model(
        L,
            layers.TokenEmbedding(sp.GetPieceSize(), 50),
        layers.Dense(sp.GetPieceSize(), activation="softmax")
    )

    embedding_test_model.compile(loss="categorical_crossentropy")

    embedding_test_model.fit(tokens, tokens, learning_rate=0.01, epochs=10, verbose=1)


    # embedder = layers.Embedding(sp.GetPieceSize(), 2)
    # embedder.init_weights(L)
    # print(embedder.get_output_shape())
    # print(embedder.get_output(tokens, len(tokens)))
    # print(sp.Decode(tokens.tolist()))

if __name__ == '__main__':
    main()