import tensorflow as tf


class Trainer(object):
    def __init__(self, name, n, obs_shape, act_space, agent_index, args):
        raise NotImplemented()

    def model_save_dict(self):
        raise NotImplemented()

    @tf.function
    def action(self, obs):
        raise NotImplemented()

    def update(self, agents, experience):
        raise NotImplemented()
