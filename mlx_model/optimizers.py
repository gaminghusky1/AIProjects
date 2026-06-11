import mlx.core as mx

def clip_grads_by_global_norm(grads, clip_norm, epsilon=1e-6):
    if clip_norm is None or len(grads) == 0:
        return grads

    total = mx.array(0.0, dtype=mx.float32)
    for g in grads:
        total = total + mx.sum(g * g)

    global_norm = mx.sqrt(total)
    scale = mx.minimum(1.0, clip_norm / (global_norm + epsilon))

    return [g * scale for g in grads]

class GradientDescent:
    def __init__(self, learning_rate=0.01, clip_norm=None):
        self.learning_rate = learning_rate
        self.clip_norm = clip_norm

    def step(self, param_refs, grads):
        params = [getattr(*param_ref) for param_ref in param_refs]
        grads = clip_grads_by_global_norm(grads, self.clip_norm)
        for i in range(len(params)):
            params[i] = params[i] - self.learning_rate * grads[i]

        for i in range(len(params)):
            setattr(param_refs[i][0], param_refs[i][1], params[i])

        mx.eval(*params)


class Adam:
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, clip_norm=None):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.clip_norm = clip_norm
        self.m = None
        self.v = None
        self.t = 0

    def step(self, param_refs, grads):
        params = [getattr(*param_ref) for param_ref in param_refs]
        grads = clip_grads_by_global_norm(grads, self.clip_norm)
        if self.m is None:
            self.m = [mx.zeros_like(p) for p in params]
            self.v = [mx.zeros_like(p) for p in params]

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] * grads[i])

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            update = self.learning_rate * m_hat / (mx.sqrt(v_hat) + self.epsilon)
            params[i] = params[i] - update

            # print("delta:", mx.mean(mx.abs(params[i] - getattr(*param_refs[i]))))
            # print("grad:", mx.mean(mx.abs(grads[i])))
            # print("Params:", sum(p.size for p in params))
            # print("Gradients:", sum(g.size for g in grads))
            # print("M:", sum(m.size for m in self.m))
            # print("V:", sum(v.size for v in self.v))

        for i in range(len(params)):
            setattr(param_refs[i][0], param_refs[i][1], params[i])

        mx.eval(*params, *self.m, *self.v)

class AdamW:
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.01, clip_norm=None):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.m = None
        self.v = None
        self.t = 0

    def step(self, param_refs, grads):
        params = [getattr(*param_ref) for param_ref in param_refs]
        grads = clip_grads_by_global_norm(grads, self.clip_norm)
        if self.m is None:
            self.m = [mx.zeros_like(p) for p in params]
            self.v = [mx.zeros_like(p) for p in params]

        self.t += 1

        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] * grads[i])

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            adam_update = m_hat / (mx.sqrt(v_hat) + self.epsilon)

            obj, name = param_refs[i]
            skip_decay = (
                    name in {"biases", "up_biases", "down_biases", "b_o", "scale", "shift"}
                    or obj.__class__.__name__ in {"TokenEmbedding", "PositionalEmbedding"}
            )

            if skip_decay:
                params[i] = params[i] - self.learning_rate * adam_update
            else:
                params[i] = params[i] * (1 - self.learning_rate * self.weight_decay) - self.learning_rate * adam_update

        for i in range(len(params)):
            setattr(param_refs[i][0], param_refs[i][1], params[i])

        mx.eval(*params, *self.m, *self.v)

optimizer_dict = {
    None: GradientDescent,
    'adam': Adam,
    'adamw': AdamW,
}