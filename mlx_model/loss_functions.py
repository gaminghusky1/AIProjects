import mlx.core as mx
import math

def mse(y, y_hat, mask=None):
    loss = mx.power(y - y_hat, 2)

    if mask is None:
        return mx.mean(loss)

    mask = mask.astype(loss.dtype)
    return mx.sum(loss * mask) / mx.sum(mask)

def mse_derivative(y, y_hat, mask=None):
    grad = 2 * (y_hat - y)

    if mask is None:
        return grad / math.prod(y.shape)

    mask = mask.astype(grad.dtype)
    while len(mask.shape) < len(grad.shape):
        mask = mask[..., mx.newaxis]

    return grad * mask / mx.sum(mask)

def categorical_crossentropy(y, y_hat, mask=None):
    y_hat = mx.clip(y_hat, 1e-12, 1 - 1e-12)
    per_item_loss = -mx.sum(y * mx.log(y_hat), axis=-1)

    if mask is None:
        return mx.mean(per_item_loss)

    return mx.sum(per_item_loss * mask) / mx.sum(mask)

def categorical_crossentropy_derivative(y, y_hat, mask=None):
    y_hat = mx.clip(y_hat, 1e-12, 1 - 1e-12)
    grad = -y / y_hat

    if mask is None:
        return grad / (math.prod(y.shape) / y.shape[-1])

    mask = mask.astype(grad.dtype)
    return (grad * mask[..., mx.newaxis]) / mx.sum(mask)

def softmax_crossentropy_derivative(y, y_hat, mask=None):
    grad = y_hat - y

    if mask is None:
        return grad / (math.prod(y.shape) / y.shape[-1])

    mask = mask.astype(grad.dtype)
    return (grad * mask[..., mx.newaxis]) / mx.sum(mask)

# Mask should be 1 where loss should be included and 0 elsewhere (ex. 1 at assistant output tokens and 0 at user input tokens)
def sparse_softmax_crossentropy(y, y_hat, mask=None):
    logits_stable = y_hat - mx.max(y_hat, axis=-1, keepdims=True)
    logsumexp = mx.log(mx.sum(mx.exp(logits_stable), axis=-1))

    b, t, v = y_hat.shape

    true_logits = logits_stable[
        mx.arange(b)[:, mx.newaxis],
        mx.arange(t)[mx.newaxis, :],
        y
    ]

    per_token_loss = logsumexp - true_logits

    if mask is None:
        return mx.mean(per_token_loss)

    mask = mask.astype(per_token_loss.dtype)
    return mx.sum(per_token_loss * mask) / mx.sum(mask)

# Mask should be 1 where loss should be included and 0 elsewhere (ex. 1 at assistant output tokens and 0 at user input tokens)
def sparse_softmax_crossentropy_derivative(y, y_hat, mask=None):
    logits_stable = y_hat - mx.max(y_hat, axis=-1, keepdims=True)
    exp_logits = mx.exp(logits_stable)
    probs = exp_logits / mx.sum(exp_logits, axis=-1, keepdims=True)

    b, t, v = y_hat.shape

    grad = probs
    grad[
        mx.arange(b)[:, mx.newaxis],
        mx.arange(t)[mx.newaxis, :],
        y
    ] -= 1.0

    if mask is None:
        return grad / (b * t)

    mask = mask.astype(grad.dtype)
    return (grad * mask[:, :, mx.newaxis]) / mx.sum(mask)

loss_func_dict = {
    'mse': [mse, mse_derivative],
    'categorical_crossentropy': [categorical_crossentropy, categorical_crossentropy_derivative],
    'softmax_crossentropy': [categorical_crossentropy, softmax_crossentropy_derivative],
    'sparse_softmax_crossentropy': [sparse_softmax_crossentropy, sparse_softmax_crossentropy_derivative],
}

class LossFunction:
    def __init__(self, func_name):
        if func_name not in loss_func_dict:
            raise ValueError(f'Invalid loss function identifier: "{func_name}"')
        func_list = loss_func_dict[func_name]
        self.func = func_list[0]
        self.func_deriv = func_list[1]

    def __call__(self, y, y_hat, mask=None):
        return self.func(y, y_hat, mask=mask)

    def derivative(self, y, y_hat, mask=None):
        return self.func_deriv(y, y_hat, mask=mask)