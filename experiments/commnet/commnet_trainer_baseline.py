from experiments.Trainer import Trainer
import numpy as np
import tensorflow as tf

from maddpg.model.commnet import CNActorController
from maddpg.common.tf_util import clipnorm, update_target


class CommnetTrainer(Trainer):
    def __init__(self, name, n, obs_shape, act_space, agent_index, args, **kwargs):

        self.name = name
        self.n = n
        self.agent_index = agent_index
        self.args = args

        self.actors = CNActorController(act_space=act_space, args=args, **kwargs)

    def model_save_dict(self):
        return {
            "actors": self.actors
        }

    @tf.function
    def action(self, obs, mem):
        p = self.actor.sample(obs, mem)
        return p[0]

    def update(self, agents, experience):
        obs_n = [tf.convert_to_tensor(i, dtype=tf.keras.backend.floatx()) for i in experience["obs_n"]]
        act_n = [tf.convert_to_tensor(i, dtype=tf.keras.backend.floatx()) for i in experience["act_n"]]
        obs_next_n = [tf.convert_to_tensor(i, dtype=tf.keras.backend.floatx()) for i in experience["obs_next_n"]]

        cct_obs, cct_act = tf.concat(obs_n, -1), tf.concat(act_n, -1)
        cct_obs_next = tf.concat(obs_next_n, -1)

        num_sample = 1
        target_q = 0.0
        target_q_next = None

        with tf.GradientTape() as tape_p:
            act, act_pd = self.actors.sample_reg(cct_obs)

            a_pd = tf.one_hot(tf.math.argmax(act_pd))*act_pd
            log_p = -tf.math.log(tf.nn.softmax(act_pd)+1e-4)*a_pd
            baseline = tf.reduce_mean(tf.sigmoid(act_pd))
            reward = experience["rew"]

            loss = log_p*(reward - baseline) + 0.03 * (reward - baseline)**2


        actor_grad = tape_p.gradient(p_loss, self.actor.trainable_weights)
        self.actor.optimizer.apply_gradients(zip(clipnorm(actor_grad, 0.5), self.actor.trainable_weights))

        return np.array([q_loss, p_loss, np.mean(target_q), np.mean(experience["rew"]), np.mean(target_q_next), np.std(target_q)])
