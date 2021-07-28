import tensorflow as tf

from maddpg.common.distributions import make_pdtype
import maddpg.common.noise_fn as nfn
from maddpg.common.tf_util import sample_soft


class CNActorController(tf.keras.Model):
    def __init__(self, act_space, n_agents=3, h_units=256, name="", args=None):
        super().__init__(name="CNActorControllerI" + name + "I")

        self.act_space = act_space
        self.act_pd = make_pdtype(act_space)
        self.pd = None

        self.NO_MASK = tf.zeros(1, dtype=tf.float32)
        self.NO_OU = self.NO_MASK, self.NO_MASK, self.NO_MASK, self.NO_MASK

        self.shape = (1, n_agents, h_units)

        self.noise_s_fn = nfn.identity
        self.noise_r_fn = nfn.generate_noise(
            shape=self.shape,
            way=nfn.NoiseNames.WAY_ADD,
            type=nfn.NoiseNames.TYPE_PROBABILITY,
            val=nfn.NoiseNames.VALUE_UNIFORM,
            pck={
                'prob': 0.2,
                'range': (-1, 1),
                'value': 2
            })

        self.n_agents = n_agents
        self.h_units = h_units
        self.c_units = h_units # Must be equal
        self.c_layers = args.communication_steps

        self.encoder = tf.keras.layers.Dense(self.h_units, tf.nn.relu, bias_initializer='random_normal')
        self.encoder_2 = tf.keras.layers.Dense(self.h_units, tf.nn.relu, bias_initializer='random_normal')

        self.gru_cell = \
            [tf.keras.layers.GRUCell(self.h_units) for _ in range(self.c_layers)]

        self.decoder = tf.keras.layers.Dense(self.h_units, tf.nn.relu, bias_initializer='random_normal')
        self.output_layer = tf.keras.layers.Dense(act_space.n, bias_initializer='random_normal')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=0.5)

        self.extra_metrics = [(None, None, None, None) for _ in range(self.c_layers)]

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
            x_with_noise = self.noise_s_fn(x, mask, (ou_s[0], ou_s[1]))
            ci = (tf.math.reduce_sum(x_with_noise, axis=1, keepdims=True)) / (self.n_agents)
            ci_with_noise = self.noise_r_fn(ci, mask, (ou_s[2], ou_s[3]))

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
            # Disabled self communication
            # in paper it is not disabled (tf.math.reduce_sum(x, axis=1, keepdims=True) / (self.n_agents))
            # here it is disabled ((tf.math.reduce_sum(x, axis=1, keepdims=True) - x) / (self.n_agents - 1))

            # Should there be a stop gradient
            x_with_noise = self.noise_s_fn(x, mask, (ou_s[0], ou_s[1]))
            ci = (tf.math.reduce_sum(x_with_noise, axis=1, keepdims=True)) / (self.n_agents)
            ci_with_noise = self.noise_r_fn(ci, mask, (ou_s[2], ou_s[3]))

            self.extra_metrics[i] = (x, x_with_noise-x, ci, ci_with_noise-ci)

            x = self.gru_cell[i](ci_with_noise, states=x)[0]
            x = x + h0

        return self.extra_metrics

    @tf.function
    def sample(self, obs):
        p = self.call((obs, self.NO_MASK, self.NO_OU))
        tf_action = sample_soft(p)
        return tf_action

    @tf.function
    def act_sample(self, obs, mask, ou_s):
        p = self.call((obs, mask, ou_s))
        tf_action = sample_soft(p)
        return tf_action

    # must already doing log(softmax(p) + bias) and mean(sigmoid)
    @tf.function
    def sample_reg(self, obs):
        p = self.call((obs, self.NO_MASK, self.NO_OU))
        tf_action = sample_soft(p)
        return tf_action, p

    @tf.function
    def raw(self, obs):
        p = self.call((obs, self.NO_MASK, self.NO_OU))
        return p


class CNActorControllerNoComm(CNActorController):
    def __init__(self, act_space, n_agents=3, h_units=256, name="", args=None):
        super().__init__( act_space, n_agents, h_units, name, args)

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