import tensorflow as tf
import numpy as np
from maddpg.common.distributions import make_pdtype


class MLP_Model(tf.keras.Model):
    def __init__(self, n_layers, n_units, activation, name=""):
        super().__init__(name="MLPI" + name + "I")
        self.n_layers = n_layers
        self.n_units = n_units
        self.activation = activation

        self._layers = []
        for _ in range(self.n_layers):
            self._layers.append(tf.keras.layers.Dense(units=self.n_units, activation=self.activation))

    @tf.function
    def call(self, inputs):
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x


class MLP2_Model(tf.keras.Model):
    def __init__(self, n_units, activation, name=""):
        super().__init__(name="MLP2I" + name + "I")
        self.n_units = n_units
        self.activation = activation

        self._layers0 = tf.keras.layers.Dense(units=self.n_units, activation=self.activation)
        self._layers1 = tf.keras.layers.Dense(units=self.n_units, activation=self.activation)

    @tf.function
    def call(self, inputs):
        x = inputs
        x = self._layers0(x)
        x = self._layers1(x)

        return x


class Actor(tf.keras.Model):
    def __init__(self, act_space, n_units=64, name=""):
        super().__init__(name="ActorI" + name + "I")
        self.base_model = MLP2_Model(n_units, tf.nn.relu, name)

        self.act_space = act_space
        self.act_pd = make_pdtype(act_space)
        self.pd = None

        self.output_layer = tf.keras.layers.Dense(act_space.n)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    @tf.function
    def call(self, inputs):
        x = inputs
        x = self.base_model(x)
        x = self.output_layer(x)
        return x

    @tf.function
    def sample(self, obs):
        p = self.call(obs)
        tf_action = self.act_pd.pdfromflat(p).sample()
        return tf_action

    @tf.function
    def sample_reg(self, obs):
        p = self.call(obs)
        pd = self.act_pd.pdfromflat(p)
        tf_action = pd.sample()
        tf_logits = pd.flatparam()
        return tf_action, tf_logits

    @tf.function
    def raw(self, obs):
        p = self.call(obs)
        return p


class Critic(tf.keras.Model):
    def __init__(self, n_units=64, name=""):
        super().__init__(name="Critic(" + name + ")")
        self.base_model = MLP2_Model(n_units, tf.nn.relu, name)

        self.output_layer = tf.keras.layers.Dense(1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    @tf.function
    def call(self, inputs):
        x = inputs
        x = self.base_model(x)
        x = self.output_layer(x)
        return x

    @tf.function
    def eval(self, obs):
        q = self.call(obs)
        return q

    @tf.function
    def eval1(self, obs):
        q = self.call(obs)
        return q

    @tf.function
    def eval2(self, obs):
        q = self.call(obs)
        return q
