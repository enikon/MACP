import tensorflow as tf

from maddpg.common.distributions import make_pdtype


class MDActorNetwork(tf.keras.Model):
    def __init__(self, act_space, name="", args=None):
        super().__init__(name="MDActorI" + name + "I")

        self.act_space = act_space
        self.act_pd = make_pdtype(act_space)
        self.pd = None

        self.output_layer = tf.keras.layers.Dense(act_space.n)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, clipnorm=0.5)

        self.encoder_units = 512

        self.read_units = 128

        # MUST BE EQUAL
        self.memory_units = args.memory_size
        self.write_units = self.memory_units

        self.action_units = 256
        self.attention_units = 128

        self.action_number = act_space.n

        self.encode_layer = tf.keras.layers.Dense(self.encoder_units, tf.nn.relu)
        self.read_projection = tf.keras.layers.Dense(self.read_units, tf.nn.relu)

        self.read_layer = tf.keras.layers.Dense(self.read_units)
        self.write_projection = tf.keras.layers.Dense(self.write_units, tf.nn.sigmoid)

        self.write_layer = tf.keras.layers.Dense(self.write_units, tf.nn.tanh)
        self.remember_layer = tf.keras.layers.Dense(self.memory_units, tf.nn.sigmoid)
        self.forget_layer = tf.keras.layers.Dense(self.memory_units, tf.nn.sigmoid)

        self.action_layer = tf.keras.layers.Dense(self.action_units, tf.nn.relu)

    @tf.function
    def call(self, *inputs):
        observation, memory = inputs

        #ENCODE
        encode_out = self.encode_layer(observation)
        obs_encode_out = self.read_projection(encode_out)

        #READ
        read_out = self.read_layer(obs_encode_out)
        concat_read_out = tf.concat([obs_encode_out, read_out, memory], axis=-1)
        read_info_out = memory * self.write_projection(concat_read_out)

        #WRITE
        concat_before_write_out = tf.concat([obs_encode_out, memory], axis=-1)
        write_out = self.write_layer(concat_before_write_out)
        memory_new_out = self.remember_layer(obs_encode_out) * write_out + self.forget_layer(obs_encode_out) * memory

        #ACTION
        concat = tf.concat([obs_encode_out, read_info_out, memory_new_out], axis=-1)
        action_out = self.action_layer(concat)
        output_out = self.output_layer(action_out)

        return output_out, memory_new_out

    @tf.function
    def sample(self, obs, mem):
        p, m = self.call(obs, mem)
        tf_action = self.act_pd.pdfromflat(p).sample()
        return tf_action, m

    @tf.function
    def sample_reg(self, obs, mem):
        p, m = self.call(obs, mem)
        pd = self.act_pd.pdfromflat(p)
        tf_action = pd.sample()
        tf_logits = pd.flatparam()
        return tf_action, m, tf_logits

    @tf.function
    def raw(self, obs, mem):
        p, m = self.call(obs, mem)
        return p, m