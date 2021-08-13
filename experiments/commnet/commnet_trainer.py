from experiments.Trainer import Trainer
import numpy as np
import tensorflow as tf

from maddpg.model.commnet import CNActorController, CNActorControllerNoComm
from maddpg.common.tf_util import clipnorm, update_target
from maddpg.model.maddpg import Critic


class CommnetTrainer(Trainer):
    def __init__(self, name, n, obs_shape, act_space, args, **kwargs):

        self.name = name
        self.n = n
        self.args = args

        self.polyak_rate_iterator = 0

        actor_network_class = CNActorControllerNoComm if args.disable_comm else CNActorController

        self.actors = actor_network_class(act_space=act_space, n_agents=n, args=args, **kwargs)
        self.target_actors = actor_network_class(act_space=act_space,  n_agents=n, args=args, **kwargs)

        self.critic = Critic(args=args)
        self.target_critic = Critic(args=args)

    def model_save_dict(self):
        return {
            "actors": self.actors,
            "target_actors": self.target_actors,
            "critic": self.critic,
            "target_critic": self.target_critic
        }

    def get_noise_shape(self):
        return self.actors.noise_shape

    @tf.function
    def action(self, obs, mask, ou_s):
        p = self.actors.act_sample(tf.expand_dims(obs, axis=0), mask, ou_s)[0]
        return p

    def update(self, agents, experience):
        obs_n = [tf.convert_to_tensor(i, dtype=tf.keras.backend.floatx()) for i in experience["obs_n"]]
        act_n = [tf.convert_to_tensor(i, dtype=tf.keras.backend.floatx()) for i in experience["act_n"]]
        obs_next_n = [tf.convert_to_tensor(i, dtype=tf.keras.backend.floatx()) for i in experience["obs_next_n"]]

        q_loss, p_loss, target_q, target_q_next = self._update(agents, experience, obs_n, act_n, obs_next_n)
        return np.array([q_loss, p_loss, np.mean(target_q), np.mean(experience["rew"]), np.mean(target_q_next), np.std(target_q)])

    @tf.function
    def _update(self, agents, experience, obs_n, act_n, obs_next_n):
        cct_obs, cct_act = tf.concat(obs_n, -1), tf.concat(act_n, -1)
        cct_obs_next = tf.concat(obs_next_n, -1)

        ustck_obs = tf.squeeze(tf.concat(tf.unstack(obs_n, axis=-2), -1))
        ustck_act = tf.squeeze(tf.concat(tf.unstack(act_n, axis=-2), -1))

        self.polyak_rate_iterator += 1
        polyak_rate = 200
        num_sample = 1
        target_q = 0.0
        target_q_next = None

        for i in range(num_sample):
            target_act_next_a_n = self.target_actors.sample(cct_obs_next)
            target_q_next = self.target_critic.eval1(
                tf.concat(
                    (ustck_obs, tf.squeeze(tf.concat(tf.unstack(target_act_next_a_n, axis=-2), -1))),
                    axis=-1
                )
            )
            target_q += experience["rew"] + self.args.gamma * (1.0 - experience["done"]) * tf.squeeze(target_q_next, axis=-1)
        target_q /= num_sample

        with tf.GradientTape() as tape_q:
            q = tf.squeeze(self.critic.eval2(tf.concat((ustck_obs, ustck_act), -1)), axis=-1)
            q_loss = tf.keras.losses.mean_squared_error(target_q, q)

        critic_grad = tape_q.gradient(q_loss, self.critic.trainable_weights)
        self.critic.optimizer.apply_gradients(zip(clipnorm(critic_grad, 0.5), self.critic.trainable_weights))

        with tf.GradientTape() as tape_p:
            act, act_pd = self.actors.sample_reg(cct_obs)
            unstck_act_p = tf.squeeze(tf.concat(tf.unstack(act, axis=-2), -1))

            q = self.critic.eval(tf.concat((ustck_obs, unstck_act_p), -1))
            p_reg = tf.reduce_mean(tf.square(act_pd))
            p_loss = (-tf.reduce_mean(q) + p_reg * 1e-4)*100

        actor_grad = tape_p.gradient(p_loss, self.actors.trainable_weights)
        #self.actors.optimizer.apply_gradients(zip(clipnorm(actor_grad, 0.5), self.actors.trainable_weights))
        self.actors.optimizer.apply_gradients(zip(actor_grad, self.actors.trainable_weights))

        if self.polyak_rate_iterator % polyak_rate == 0:
            update_target(self.actors, self.target_actors)
            update_target(self.critic, self.target_critic)

        return q_loss, p_loss, target_q, target_q_next
