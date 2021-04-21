import collections
import numpy as np
import os
import tensorflow as tf

def sum(x, axis=None, keepdims=False):
    return tf.reduce_sum(x, axis=None if axis is None else [axis], keep_dims = keepdims)
def mean(x, axis=None, keepdims=False):
    return tf.reduce_mean(x, axis=None if axis is None else [axis], keep_dims = keepdims)
def var(x, axis=None, keepdims=False):
    meanx = mean(x, axis=axis, keepdims=keepdims)
    return mean(tf.square(x - meanx), axis=axis, keepdims=keepdims)
def std(x, axis=None, keepdims=False):
    return tf.sqrt(var(x, axis=axis, keepdims=keepdims))
def max(x, axis=None, keepdims=False):
    return tf.reduce_max(x, axis=None if axis is None else [axis], keep_dims = keepdims)
def min(x, axis=None, keepdims=False):
    return tf.reduce_min(x, axis=None if axis is None else [axis], keep_dims = keepdims)
def concatenate(arrs, axis=0):
    return tf.concat(axis=axis, values=arrs)
def argmax(x, axis=None):
    return tf.argmax(x, axis=axis)
def softmax(x, axis=None):
    return tf.nn.softmax(x, axis=axis)

#############
# Functions #
#############


def clipnorm(grad, norm=0.5):
    return [tf.clip_by_norm(g,norm) for g in grad]


def update_target(model, target_model):
    polyak = 1.0 - 1e-2
    new_weight = np.array(model.get_weights(), dtype=object)
    old_weight = np.array(target_model.get_weights(), dtype=object)
    target_model.set_weights(polyak * old_weight + (1 - polyak) * new_weight)


###################
# Troubleshooting #
###################


def allow_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    return

