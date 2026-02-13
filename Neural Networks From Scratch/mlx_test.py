import time
import mlx.core as mx
import numpy as np

n = 8192
a = mx.random.normal((n, n), dtype=mx.float32)
b = mx.random.normal((n, n), dtype=mx.float32)

mx.eval(a @ b)  # warmup

t0 = time.time()
y = a @ b
mx.eval(y)
print("matmul seconds:", time.time() - t0)

a = np.random.randn(n, n)
b = np.random.randn(n, n)

t0 = time.time()
y = a @ b
print("numpy matmul seconds:", time.time() - t0)