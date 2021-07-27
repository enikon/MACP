from enum import Enum

import tensorflow as tf


@tf.function
def identity(x, m, ou_s):
    return x


class NoiseNames:
    WAY_ADD = 0
    WAY_MUL = 1
    WAY_REP = 2
    TYPE_ALL = 0
    TYPE_PROBABILITY = 1
    VALUE_UNIFORM = 0
    VALUE_CONSTANT = 1


def generate_noise(shape,
                   way=NoiseNames.WAY_ADD,
                   type=NoiseNames.TYPE_ALL,
                   val=NoiseNames.VALUE_UNIFORM,
                   pck=None):

    if pck is None:
        pck = {'range': (-1, 1), 'prob': 0.1, 'value': 1}

    #############
    # OPERATION #
    #############

    op = None
    if way == NoiseNames.WAY_ADD:
        @tf.function
        def noise_op_add(x, g, y, m):
            return x + tf.stop_gradient(m * g * y)
        op = noise_op_add
    elif way == NoiseNames.WAY_MUL:
        @tf.function
        def noise_op_mul(x, g, y, m):
            return x * tf.stop_gradient((m * g * y) + (1 - m * g))
        op = noise_op_mul
    elif way == NoiseNames.WAY_REP:
        @tf.function
        def noise_op_mul(x, g, y, m):
            return x * tf.stop_gradient((1 - m * g)) + tf.stop_gradient(m * g * y)

        op = noise_op_mul

    #############
    # NOISE GEN #
    #############

    gen = None
    if type == NoiseNames.TYPE_ALL:
        @tf.function
        def gen_all(rand):
            return tf.ones(shape)
        gen = gen_all
    elif type == NoiseNames.TYPE_PROBABILITY:
        @tf.function
        def gen_choose(rand):
            return (1 - tf.sign(tf.sign(
                tf.clip_by_value(rand, 0, 1) - pck['prob']
            ) + 0.5))/2
        gen = gen_choose

    #########
    # VALUE #
    #########

    value = None
    if val == NoiseNames.VALUE_UNIFORM:
        @tf.function
        def val_random(rand):
            return tf.clip_by_value(rand, 0, 1) * (pck['range'][1] - pck['range'][0]) + pck['range'][0]
        value = val_random
    elif val == NoiseNames.VALUE_CONSTANT:
        @tf.function
        def val_constant(rand):
            return tf.ones(shape=shape) * pck['value']
        value = val_constant

    ################
    # FINALISATION #
    ################

    @tf.function
    def custom_noise(x, m, ou_s):
        return op(x, gen(ou_s[0]), value(ou_s[1]), m)

    return custom_noise


class NoiseOUManager(object):
    def __init__(self, noise_list, noise_assignment):
        sgi, svi, rgi, rvi = noise_assignment
        self.noise_get = noise_list[sgi], noise_list[svi], noise_list[rgi], noise_list[rvi]
        self.noise_list = noise_list

    def get(self):
        return self.noise_get[0].state, self.noise_get[1].state, self.noise_get[2].state,self.noise_get[3].state

    def reset(self):
        for i in self.noise_list:
            i.reset()
        return self.get()

    def update(self):
        for i in self.noise_list:
            i.update()
        return self.get()


class NoiseOU(object):
    def __init__(self, shape, alpha):
        self.shape = shape
        self.alpha = alpha
        self.state = None

    def reset(self):
        self.state = tf.random.uniform(self.shape)

    def update(self):
        self.state = self.state + self.alpha * tf.random.uniform(self.shape, minval=-1, maxval=1)


class NoiseUniform(object):
    def __init__(self, shape):
        self.shape = shape
        self.state = None

    def reset(self):
        self.state = tf.random.uniform(self.shape)

    def update(self):
        self.state = tf.random.uniform(self.shape)
