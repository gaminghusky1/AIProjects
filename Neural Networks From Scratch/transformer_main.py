import numpy as np
import layers
import model
import masks
import sentencepiece as spm
import pickle
# np.seterr(all="raise")

SPECIAL_TOKENS = [
    ">newconversation<",
    ">persona<",
    ">personb<",
    ">endoftext<",
]

def load_data(
    filename: str,
    sp_model_path: str = "TransformerData/spm_dialogue.model",
    block_size: int = 128,
    stride: int | None = None,
    max_tokens: int | None = None,
    add_bos: bool = False,
    add_eos: bool = False,
    verify_special_tokens: bool = True,
    one_hot_labels: bool = True,
):
    if stride is None:
        stride = block_size

    # 1) Read text
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()

    # 2) Load SentencePiece
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)
    vocab_size = sp.GetPieceSize()

    # Optional verification
    if verify_special_tokens:
        for tok in SPECIAL_TOKENS:
            if sp.PieceToId(tok) == sp.unk_id():
                raise ValueError(f"{tok} missing from tokenizer vocab.")

    # 3) Encode
    ids = sp.EncodeAsIds(text)

    if add_bos:
        ids = [sp.bos_id()] + ids
    if add_eos:
        ids = ids + [sp.eos_id()]

    if max_tokens is not None:
        ids = ids[:max_tokens]

    ids = np.asarray(ids, dtype=np.int64)

    L = block_size + 1
    n_ids = ids.shape[0]
    if n_ids < L:
        raise ValueError("Not enough tokens.")

    # 4) Build windows
    starts = np.arange(0, n_ids - L + 1, stride, dtype=np.int64)
    N = len(starts)

    windows = np.empty((N, L), dtype=np.int64)
    for i, s in enumerate(starts):
        windows[i] = ids[s:s+L]

    x = windows[:, :-1]   # (N, block_size)
    y = windows[:,  1:]   # (N, block_size)

    # 5) Optional one-hot encode y
    if one_hot_labels:
        y_onehot = np.zeros((N, block_size, vocab_size), dtype=np.float32)

        # Efficient vectorized indexing
        rows = np.arange(N)[:, None]
        cols = np.arange(block_size)[None, :]
        y_onehot[rows, cols, y] = 1.0

        y = y_onehot

    return x, y, vocab_size

def main():
    block_size = 128
    train_data, test_data, vocab_size = load_data("TransformerData/conversations.txt", block_size=block_size)
    d_model = 512
    d_ff = 4 * d_model
    transformer_model = model.Model(
        (block_size,),
        layers.TokenEmbedding(vocab_size, d_model),
        layers.PositionalEmbedding(),

        layers.ResidualBlock(
            layers.LayerNorm(),
            layers.Attention(d_model, d_model, heads=8, mask=masks.causal),
        ),
        layers.ResidualBlock(
            layers.LayerNorm(),
            layers.MultilayerPerceptron(d_ff, activation='relu')
        ),

        layers.ResidualBlock(
            layers.LayerNorm(),
            layers.Attention(d_model, d_model, heads=8, mask=masks.causal),
        ),
        layers.ResidualBlock(
            layers.LayerNorm(),
            layers.MultilayerPerceptron(d_ff, activation='relu')
        ),

        layers.ResidualBlock(
            layers.LayerNorm(),
            layers.Attention(d_model, d_model, heads=8, mask=masks.causal),
        ),
        layers.ResidualBlock(
            layers.LayerNorm(),
            layers.MultilayerPerceptron(d_ff, activation='relu')
        ),

        layers.LayerNorm(),
        layers.TimeDistributedDense(vocab_size, activation='crossentropy_softmax')
    )

    transformer_model.compile(loss='softmax_crossentropy')

    transformer_model.fit(train_data, test_data, epochs=10, learning_rate=0.005, batch_size=1, verbose=2)

    transformer_model.save_as("Models/transformer_test")

if __name__ == "__main__":
    main()