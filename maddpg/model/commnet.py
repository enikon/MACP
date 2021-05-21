import tensorflow as tf

from maddpg.common.distributions import make_pdtype


class CNActorController(tf.keras.Model):
    def __init__(self, act_space, n_agents=3, h_units=32, name="", args=None):
        super().__init__(name="CNActorControllerI" + name + "I")

        self.act_space = act_space
        self.act_pd = make_pdtype(act_space)
        self.pd = None

        self.n_agents = n_agents
        self.h_units = h_units
        self.c_units = h_units # Must be equal
        self.c_layers = args.communication_steps
        self.node_activation = tf.nn.tanh

        self.encoder = tf.keras.layers.Dense(self.h_units, tf.nn.relu)
        self.H = [tf.keras.layers.Dense(self.h_units) for _ in range(self.c_layers)]
        self.C = [tf.keras.layers.Dense(self.c_units) for _ in range(self.c_layers)]
        self.output_layer = tf.keras.layers.Dense(act_space.n)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=0.5)

    @tf.function
    def call(self, inputs):
        x = inputs
        x = self.encoder(x)
        for H, C in zip(self.H, self.C):
            hi = H(x)
            ci = C(x)
            x = self.node_activation(hi + (tf.reduce_sum(ci) - ci)/(self.num_agents-1))
        x = self.decoder(x)
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
        # must do already doing log(softmax(p) + bias) and mean(sigmoid)
        pd = self.act_pd.pdfromflat(p)
        tf_action = pd.sample()
        tf_logits = pd.flatparam() #==p
        return tf_action, tf_logits

    @tf.function
    def raw(self, obs):
        p = self.call(obs)
        return p