import numpy as np
import mlx.core as mx


def main():
    a = np.array([
        [1, 1],
        [2, 2]
    ])

    # b = np.array([
    #     [1, 2],
    #     [3, 4],
    #     [5, 6]
    # ])

    b = np.array([
        [[5, 7],
         [7, 8]],
        [[5, 7],
         [7, 8]]
    ])

    # sm_deriv_a = activations.transformer_softmax_derivative(a)

    # print(np.einsum('bts,btss->bts', a, sm_deriv_a))

    # print(np.einsum('bsv,btv->ts', a, b))
    # b, n, m = a.shape
    # diag_indices = np.arange(m)
    # diags = np.zeros((b, n, m, m))
    # diags[:, :, diag_indices, diag_indices] = a
    # # jacobians
    # # print(diags)
    # print(diags - np.einsum('bni,bnj->bnij', a, a))

    # print(np.einsum('btk,ik->bti', a, b))

    # print(np.einsum('bsv,btv->bst', a, b))
    # x = mx.array([
    #     [
    #         [[1, 1],
    #          [1, 1]],
    #         [[1, 1],
    #          [1, 1]]
    #     ],
    #     [
    #         [[1, 1],
    #          [1, 1]],
    #         [[1, 1],
    #          [1, 1]]
    #     ]
    # ])
    # print(x.shape)
    # mask = mx.tril(mx.ones((2, 2), dtype=mx.bool_), k=0)
    # x = mx.where(mask, x, -1e9)
    # print(x)
    print(np.einsum('bo,boj->bj', a, b))

if __name__ == '__main__':
    main()