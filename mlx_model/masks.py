import mlx.core as mx

def causal(sequence_len):
    # Upper triangle of -inf so softmax will not let k_i attend to q_j for i > j (dimension is j*i).
    causal_mask = mx.tril(mx.ones((sequence_len, sequence_len), dtype=mx.bool_), k=0)
    return causal_mask