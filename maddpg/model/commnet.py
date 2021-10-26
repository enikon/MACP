import tensorflow as tf
import numpy as np

from maddpg.common.distributions import make_pdtype
from maddpg.common.tf_util import sample_soft

class CNActorController(tf.keras.Model):
    def __init__(self, act_space, n_agents, h_units=256, name="", args=None, **kwargs):
        super().__init__(name="CNActorControllerI" + name + "I")

        self.act_space = act_space
        self.act_pd = make_pdtype(act_space)
        self.pd = None

        self.c_layers = args.communication_steps

        self.NO_MASK = tf.zeros(1, dtype=tf.float32)
        self.NO_OU = \
            [self.NO_MASK] * self.c_layers,\
            [self.NO_MASK] * self.c_layers,\
            [self.NO_MASK] * self.c_layers,\
            [self.NO_MASK] * self.c_layers

        self.noise_shape = (1, n_agents, h_units)

        self.noise_s_fn, self.noise_s_metric = kwargs['noise_s_fn']
        self.noise_r_fn, self.noise_r_metric = kwargs['noise_r_fn']

        self.n_agents = n_agents
        self.h_units = h_units
        self.c_units = h_units # Must be equal

        self.encoder = tf.keras.layers.Dense(self.h_units, tf.nn.relu, bias_initializer='random_normal')
        self.encoder_2 = tf.keras.layers.Dense(self.h_units, tf.nn.relu, bias_initializer='random_normal')

        self.gru_cell = \
            [tf.keras.layers.GRUCell(self.h_units) for _ in range(self.c_layers)]

        self.decoder = tf.keras.layers.Dense(self.h_units, tf.nn.relu, bias_initializer='random_normal')
        self.output_layer = tf.keras.layers.Dense(act_space.n, bias_initializer='random_normal')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=0.5)

        self.extra_metrics = np.empty((4, self.c_layers), dtype=object)

    @tf.function
    def call(self, inputs):
        x, mask, ou_s = inputs

        x = self.encoder(x)
        h0 = self.encoder_2(x)
        x = h0

        for i in range(self.c_layers):

            # Disabled self communication
            # in paper it is not disabled (tf.math.reduce_sum(x, axis=1, keepdims=True) / (self.n_agents))
            # here it is disabled ((tf.math.reduce_sum(x, axis=1, keepdims=True) - x) / (self.n_agents - 1))

            # Should there be a stop gradient
            x_with_noise = self.noise_s_fn(x, mask, (ou_s[0][i], ou_s[1][i]))
            ci = (tf.math.reduce_sum(x_with_noise, axis=1, keepdims=True)) / (self.n_agents)
            ci_with_noise = self.noise_r_fn(ci, mask, (ou_s[2][i], ou_s[3][i]))

            x = self.gru_cell[i](ci_with_noise, states=x)[0]
            x = x + h0

        x = self.decoder(x)
        x = self.output_layer(x)
        return x

    def metrics_call(self, inputs):
        x, mask, ou_s = inputs
        x = self.encoder(x)
        h0 = self.encoder_2(x)
        x = h0

        for i in range(self.c_layers):

            x_with_noise = self.noise_s_fn(x, mask, (ou_s[0][i], ou_s[1][i]))
            ci = (tf.math.reduce_sum(x_with_noise, axis=1, keepdims=True)) / (self.n_agents)
            ci_with_noise = self.noise_r_fn(ci, mask, (ou_s[2][i], ou_s[3][i]))

            self.extra_metrics[0, i] = x
            self.extra_metrics[1, i] = (x_with_noise - x) + mask * 0
            self.extra_metrics[2, i] = ci + mask * 0
            self.extra_metrics[3, i] = (ci_with_noise - ci) + mask * 0

            x = self.gru_cell[i](ci_with_noise, states=x)[0]
            x = x + h0

        return self.extra_metrics

    @tf.function
    def sample(self, obs, mask):
        p = self.call((obs, self.NO_MASK, self.NO_OU))
        #p = self.call((obs, mask, self.NO_OU))
        tf_action = sample_soft(p)
        return tf_action

    @tf.function
    def integ_sample(self, obs, mask, ou_s):
        p = self.call((obs, mask, ou_s))
        # p = self.call((obs, mask, self.NO_OU))
        tf_action = sample_soft(p)
        return tf_action

    @tf.function
    def act_sample(self, obs, mask, ou_s):
        p = self.call((obs, mask, ou_s))
        tf_action = sample_soft(p)
        return tf_action

    # must already doing log(softmax(p) + bias) and mean(sigmoid)
    @tf.function
    def sample_reg(self, obs, mask):
        p = self.call((obs, self.NO_MASK, self.NO_OU))
        tf_action = sample_soft(p)
        return tf_action, p

    @tf.function
    def integ_sample_reg(self, obs, mask, ou_s):
        p = self.call((obs, mask, ou_s))
        tf_action = sample_soft(p)
        return tf_action, p

    @tf.function
    def raw(self, obs):
        p = self.call((obs, self.NO_MASK, self.NO_OU))
        return p


class CNActorControllerNoComm(CNActorController):
    def __init__(self, act_space, n_agents=3, h_units=256, name="", args=None, **kwargs):
        super().__init__( act_space, n_agents, h_units, name, args, **kwargs)

    @tf.function
    def call(self, inputs):
        x, mask, ou_s = inputs
        x = self.encoder(x)
        h0 = self.encoder_2(x)
        x = h0

        for i in range(self.c_layers):
            ci = (tf.math.reduce_sum(x, axis=1, keepdims=True)) / (self.n_agents) * 0.0
            x = self.gru_cell[i](ci, x)[0]
            x = x + h0  # skip connection

        x = self.decoder(x)
        x = self.output_layer(x)
        return x


class CNActorControllerAdapting(CNActorController):
    def __init__(self, act_space, n_agents=3, h_units=256, name="", args=None, **kwargs):
        super().__init__( act_space, n_agents, h_units, name, args, **kwargs)
        self.adapting = kwargs['noise_adapting']
        self.NO_OU = (
            (self.NO_MASK, self.NO_MASK) if self.adapting[0] else self.NO_MASK,
            (self.NO_MASK, self.NO_MASK) if self.adapting[1] else self.NO_MASK,
            (self.NO_MASK, self.NO_MASK) if self.adapting[2] else self.NO_MASK,
            (self.NO_MASK, self.NO_MASK) if self.adapting[3] else self.NO_MASK
        )

class CommNetNoTanh(CNActorController):
    def __init__(self, **kwargs):
        super(CommNetNoTanh, self).__init__(**kwargs)
        self.gru_cell = \
            [tf.keras.layers.GRUCell(self.h_units, activation=tf.nn.relu) for _ in range(self.c_layers)]
            #[tf.keras.layers.GRUCell(self.h_units, activation=None) for _ in range(self.c_layers)]

class CommNetScalar(CNActorController):
    def __init__(self, **kwargs):
        super(CommNetScalar, self).__init__(**kwargs)
        self.intensifier = [tf.keras.layers.Dense(1, None, bias_initializer='random_normal') for _ in range(self.c_layers)]
        self.encoder_2 = tf.keras.layers.Dense(self.h_units, tf.nn.tanh, bias_initializer='random_normal')

    @tf.function
    def call(self, inputs):
        x, mask, ou_s = inputs
        x = self.encoder(x)
        h0 = self.encoder_2(x)
        x = h0

        newx = x
        a = self.intensifier[0](x)
        x = newx * a

        for i in range(self.c_layers-1):

            # Should there be a stop gradient
            x_with_noise = self.noise_s_fn(x, mask, (ou_s[0][i], ou_s[1][i]))
            ci = (tf.math.reduce_sum(x_with_noise, axis=1, keepdims=True)) / (self.n_agents)
            ci_with_noise = self.noise_r_fn(ci, mask, (ou_s[2][i], ou_s[3][i]))

            newx = self.gru_cell[i](tf.concat([ci_with_noise + x*0, h0], -1), states=x)[0]
            #newx = self.gru_cell[i](ci_with_noise, states=x)[0]+h0

            a = self.intensifier[i+1](tf.concat([x, h0, ci_with_noise + x * 0], -1))
            x = newx * a

        # Last
        i = self.c_layers-1
        x_with_noise = self.noise_s_fn(x, mask, (ou_s[0][i], ou_s[1][i]))
        ci = (tf.math.reduce_sum(x_with_noise, axis=1, keepdims=True)) / (self.n_agents)
        ci_with_noise = self.noise_r_fn(ci, mask, (ou_s[2][i], ou_s[3][i]))

        x = self.gru_cell[i](tf.concat([ci_with_noise + x*0, h0], -1), states=x)[0]
        # x = self.gru_cell[i](ci_with_noise, states=x)[0]+h0

        x = self.decoder(x)
        x = self.output_layer(x)
        return x

    def metrics_call(self, inputs):
        x, mask, ou_s = inputs
        x = self.encoder(x)
        h0 = self.encoder_2(x)
        x = h0

        newx = x
        a = self.intensifier[0](x)
        x = newx * a

        for i in range(self.c_layers-1):

            # Should there be a stop gradient
            x_with_noise = self.noise_s_fn(x, mask, (ou_s[0][i], ou_s[1][i]))
            ci = (tf.math.reduce_sum(x_with_noise, axis=1, keepdims=True)) / (self.n_agents)
            ci_with_noise = self.noise_r_fn(ci, mask, (ou_s[2][i], ou_s[3][i]))

            self.extra_metrics[0, i] = x
            self.extra_metrics[1, i] = (x_with_noise - x) + mask * 0
            self.extra_metrics[2, i] = ci + mask * 0
            self.extra_metrics[3, i] = (ci_with_noise - ci) + mask * 0

            x = self.gru_cell[i](tf.concat([ci_with_noise + x*0, h0], -1), states=x)[0]
            # x = self.gru_cell[i](ci_with_noise, states=x)[0] + h0

            a = self.intensifier[i+1](tf.concat([x, h0, ci_with_noise + x * 0], -1))
            x = newx * a

        # Last
        i = self.c_layers-1
        x_with_noise = self.noise_s_fn(x, mask, (ou_s[0][i], ou_s[1][i]))
        ci = (tf.math.reduce_sum(x_with_noise, axis=1, keepdims=True)) / (self.n_agents)
        ci_with_noise = self.noise_r_fn(ci, mask, (ou_s[2][i], ou_s[3][i]))

        self.extra_metrics[0, i] = x
        self.extra_metrics[1, i] = (x_with_noise - x) + mask * 0
        self.extra_metrics[2, i] = ci + mask * 0
        self.extra_metrics[3, i] = (ci_with_noise - ci) + mask * 0

        return self.extra_metrics


class CommNetVector(CommNetScalar):
    def __init__(self, **kwargs):
        super(CommNetScalar, self).__init__(**kwargs)
        self.intensifier = [tf.keras.layers.Dense(self.h_units, None, bias_initializer='random_normal') for _ in
                            range(self.c_layers)]
