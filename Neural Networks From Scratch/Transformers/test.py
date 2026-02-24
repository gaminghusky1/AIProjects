import mlx.core as mx

a = mx.array([1, 1], dtype=mx.float32)

b = mx.array([[1, 2],
              [3, 4]], dtype=mx.float32)
print(mx.einsum("i,ij->j", a, b))