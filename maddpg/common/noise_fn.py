import tensorflow as tf


@tf.function
def op_identity(x, m, ou_s):
    return x


@tf.function
def metric_identity(x, m, ou_s):
    return x*0


identity = op_identity, metric_identity


class NoiseNames:
    WAY_ADD = 0
    WAY_MUL = 1
    WAY_REP = 2
    TYPE_ALL = 0
    TYPE_PROBABILITY = 1
    TYPE_VARIABLE = 2
    VALUE_UNIFORM = 0
    VALUE_CONSTANT = 1
    VALUE_VARIABLE = 2


def generate_noise(way=NoiseNames.WAY_ADD,
                   type=NoiseNames.TYPE_ALL,
                   val=NoiseNames.VALUE_UNIFORM,
                   pck=None):

    if pck is None:
        pck = {'range': (-1, 1), 'prob': 0.1, 'value': 1}

    #############
    # OPERATION #
    #############

    op = None
    metric = None

    if way == NoiseNames.WAY_ADD:
        @tf.function
        def noise_op_add(x, g, y, m):
            return x + tf.stop_gradient(m * g * y)
        op = noise_op_add

        def noise_metric_add(x, g, y, m):
            return (m * g * y) + x*0
        metric = noise_metric_add

    elif way == NoiseNames.WAY_MUL:
        @tf.function
        def noise_op_mul(x, g, y, m):
            return x * tf.stop_gradient((m * g * y) + (1 - m * g))
        op = noise_op_mul

        def noise_metric_mul(x, g, y, m):
            return ((m * g * y) + (1 - m * g)) + x*0
        metric = noise_metric_mul

    elif way == NoiseNames.WAY_REP:
        @tf.function
        def noise_op_rep(x, g, y, m):
            return x * tf.stop_gradient((1. - m * g)) + tf.stop_gradient((m * g * y))
        op = noise_op_rep

        def noise_metric_rep(x, g, y, m):
            return (m * g * y) + x*0
        metric = noise_metric_rep

    #############
    # NOISE GEN #
    #############

    gen = None
    if type == NoiseNames.TYPE_ALL:
        @tf.function
        def gen_all(rand):
            return rand*0. + 1.
        gen = gen_all
    elif type == NoiseNames.TYPE_PROBABILITY:
        @tf.function
        def gen_choose(rand):
            return (1. - tf.sign(tf.sign(
                tf.clip_by_value(rand, 0., 1.) - pck['prob']
            ) + 0.5))/2.
        gen = gen_choose
    elif type == NoiseNames.TYPE_VARIABLE:
        @tf.function
        def gen_var(rand):
            rand_val, prob = rand
            return (1. - tf.sign(tf.sign(
                tf.clip_by_value(rand_val, 0., 1.) - prob
            ) + 0.5)) / 2.

        gen = gen_var

    #########
    # VALUE #
    #########

    value = None
    if val == NoiseNames.VALUE_UNIFORM:
        @tf.function
        def val_random(rand):
            return tf.clip_by_value(rand, 0., 1.) * (pck['range'][1] - pck['range'][0]) + pck['range'][0]
        value = val_random
    elif val == NoiseNames.VALUE_CONSTANT:
        @tf.function
        def val_constant(rand):
            return rand*0. + pck['value']
        value = val_constant
    elif val == NoiseNames.VALUE_VARIABLE:
        @tf.function
        def val_var(rand):
            rand_val, intensity = rand
            return (tf.clip_by_value(rand_val, 0, 1) * 2. - 1.) * intensity
        value = val_var

    ################
    # FINALISATION #
    ################

    @tf.function
    def custom_noise(x, m, ou_s):
        gen_ou = gen(ou_s[0])
        value_ou = value(ou_s[1])
        return op(x, tf.stop_gradient(gen_ou), tf.stop_gradient(value_ou), m)

    def metric_noise(x, m, ou_s):
        return metric(x, gen(ou_s[0]), value(ou_s[1]), m)

    return custom_noise, metric_noise


class NoiseOUManager(object):
    def __init__(self, noise_list, noise_assignment):
        sgi, svi, rgi, rvi = noise_assignment
        self.noise_get = noise_list[sgi], noise_list[svi], noise_list[rgi], noise_list[rvi]
        self.noise_list = noise_list

    def get(self):
        return self.noise_get[0].state, self.noise_get[1].state, self.noise_get[2].state, self.noise_get[3].state

    def reset(self):
        for i in self.noise_list:
            i.reset()
        return self.get()

    def update(self):
        for i in self.noise_list:
            i.update()
        return self.get()


class NoiseOU(object):
    def __init__(self, shape, alpha, init_range):
        self.shape = shape
        self.alpha = alpha
        self.state = None
        self.range = init_range

    def reset(self):
        self.state = [tf.random.uniform(i, minval=self.range[0], maxval=self.range[1]) for i in self.shape]

    def update(self):
        self.state = [
            self.state[i] + self.alpha * tf.random.uniform(self.shape[i], minval=-1, maxval=1) for i in len(self.state)
        ]


class NoiseUniform(object):
    def __init__(self, shape):
        self.shape = shape
        self.state = None

    def reset(self):
        self.state = [tf.random.uniform(i) for i in self.shape]

    def update(self):
        self.state = [tf.random.uniform(i) for i in self.shape]


class NoiseManagerOUNoCorrelation(NoiseOUManager):
    def __init__(self, shape):
        super().__init__(
            [
                NoiseUniform(shape),
                NoiseUniform(shape),
                NoiseUniform(shape),
                NoiseUniform(shape)
            ],
            [0, 1, 2, 3]
        )