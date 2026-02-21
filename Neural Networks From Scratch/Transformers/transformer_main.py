import numpy as np
from base_model import *
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
    block_size = 32
    x_train, y_train, vocab_size = load_data("TransformerData/conversations.txt", block_size=block_size)
    # sp = spm.SentencePieceProcessor()
    # sp.Load("TransformerData/spm_openwebtext_wikitext.model")
    # vocab_size = sp.GetPieceSize()
    # x_train = np.load(f"TransformerData/x_train_T{block_size}.npy")
    # y_train = np.load(f"TransformerData/y_train_T{block_size}.npy")
    # x_test = np.load(f"TransformerData/x_val_T{block_size}.npy")
    # y_test = np.load(f"TransformerData/y_val_T{block_size}.npy")
    d_model = 512
    d_ff = 4 * d_model
    head_count = 8
    transformer_model = model.Model(
        (block_size,),
        layers.TokenEmbedding(vocab_size, d_model),
        layers.PositionalEmbedding(),

        layers.ResidualBlock(
            layers.LayerNorm(),
            layers.Attention(d_model, d_model, heads=head_count, mask=masks.causal),
        ),
        layers.ResidualBlock(
            layers.LayerNorm(),
            layers.MultilayerPerceptron(d_ff, activation='gelu')
        ),

        layers.ResidualBlock(
            layers.LayerNorm(),
            layers.Attention(d_model, d_model, heads=head_count, mask=masks.causal),
        ),
        layers.ResidualBlock(
            layers.LayerNorm(),
            layers.MultilayerPerceptron(d_ff, activation='gelu')
        ),

        layers.ResidualBlock(
            layers.LayerNorm(),
            layers.Attention(d_model, d_model, heads=head_count, mask=masks.causal),
        ),
        layers.ResidualBlock(
            layers.LayerNorm(),
            layers.MultilayerPerceptron(d_ff, activation='gelu')
        ),

        layers.LayerNorm(),
        layers.TimeDistributedDense(vocab_size, activation='crossentropy_softmax')
    )

    transformer_model.compile(loss='softmax_crossentropy', optimizer='adam')

    transformer_model.fit(x_train, y_train, epochs=20, learning_rate=0.005, batch_size=1, verbose=2, y_ohe=True, save_after_each_epoch=False, path="Models/transformer_model_20_epochs")

    # print("Accuracy on test dataset:", transformer_model.test(x_test, y_test, y_ohe=False))

    transformer_model.save_as("Models/transformer_model_20_epochs")

if __name__ == "__main__":
    main()