from enum import Enum

import tensorflow as tf


@tf.function
def identity(x):
    return x


class NoiseNames:
    WAY_ADD = 0
    WAY_MUL = 1
    TYPE_ALL = 0
    TYPE_PROBABILITY = 1
    VALUE_UNIFORM = 0
    VALUE_CONSTANT = 1


def generate_noise(shape, way=NoiseNames.WAY_ADD, type=NoiseNames.TYPE_ALL, val=NoiseNames.VALUE_UNIFORM, pck=None):

    if pck is None:
        pck = {'range': (-1, 1), 'prob': 0.1, 'value': 1}

    #############
    # OPERATION #
    #############

    op = None
    if way == NoiseNames.WAY_ADD:
        @tf.function
        def noise_op_add(x, y):
            return x + tf.stop_gradient(y)
        op = noise_op_add
    elif way == NoiseNames.WAY_MUL:
        @tf.function
        def noise_op_mul(x, y):
            return x * tf.stop_gradient(y)
        op = noise_op_mul

    #############
    # NOISE GEN #
    #############

    gen = None
    if type == NoiseNames.TYPE_ALL:
        @tf.function
        def all():
            return tf.ones(shape)
        gen = all
    elif type == NoiseNames.TYPE_PROBABILITY:
        @tf.function
        def discrete():
            return (1 - tf.sign(tf.sign(tf.random.uniform(shape=shape) - pck['prob']) + 0.1))/2
        gen = discrete

    #########
    # VALUE #
    #########

    value = None
    if val == NoiseNames.VALUE_UNIFORM:
        @tf.function
        def uniform():
            return tf.random.uniform(shape=shape, minval=pck['range'][0], maxval=pck['range'][1])
        value = uniform
    elif val == NoiseNames.VALUE_CONSTANT:
        @tf.function
        def constant():
            return tf.ones(shape=shape) * pck['value']
        value = constant

    ################
    # FINALISATION #
    ################

    @tf.function
    def custom_noise(x):
        return op(x, gen() * value())

    return custom_noise




