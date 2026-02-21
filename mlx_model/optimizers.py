import mlx.core as mx

class Adam:
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, param_refs, grads):
        params = [getattr(*param_ref) for param_ref in param_refs]
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

        for i in range(len(params)):
            setattr(param_refs[i][0], param_refs[i][1], params[i])

        mx.eval(*params, *self.m, *self.v)

optimizer_dict = {
    'adam': Adam,
}