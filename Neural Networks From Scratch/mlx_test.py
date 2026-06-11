import time
import mlx.core as mx
import numpy as np

y = mx.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
])

y_hat = mx.array([
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ],
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ],
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ]
])

print(
    y_hat[
        mx.arange(3)[:, mx.newaxis],
        mx.arange(4)[mx.newaxis, :],
        y
    ]
)

# n = 8192
# a = mx.random.normal((n, n), dtype=mx.float32)
# b = mx.random.normal((n, n), dtype=mx.float32)
#
# mx.eval(a @ b)  # warmup
#
# t0 = time.time()
# y = a @ b
# mx.eval(y)
# print("matmul seconds:", time.time() - t0)
#
# a = np.random.randn(n, n)
# b = np.random.randn(n, n)
#
# t0 = time.time()
# y = a @ b
# print("numpy matmul seconds:", time.time() - t0)