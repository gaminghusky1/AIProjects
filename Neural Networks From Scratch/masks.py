import numpy as np

def causal(sequence_len):
    # Upper triangle of -inf so softmax will not let k_i attend to q_j for i > j (dimension is j*i).
    causal_mask = np.triu(np.ones((sequence_len, sequence_len)), k=1) * -1e9
    return causal_mask