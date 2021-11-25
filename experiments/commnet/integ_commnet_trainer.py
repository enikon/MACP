from experiments.Trainer import Trainer
import numpy as np
import tensorflow as tf

from experiments.commnet.commnet_trainer import CommnetTrainer
from maddpg.model.commnet import CNActorController, CNActorControllerNoComm, CNActorControllerAdapting, CommNetNoTanh, \
    CommNetScalar, CommNetVector
from maddpg.common.tf_util import clipnorm, update_target
from maddpg.model.maddpg import Critic


class IntegCommnetTrainer(CommnetTrainer):

    def update(self, agents, experience):
        mask = experience["mask"]
        obs_n = [tf.convert_to_tensor(i, dtype=tf.keras.backend.floatx()) for i in experience["obs_n"]]
        act_n = [tf.convert_to_tensor(i, dtype=tf.keras.backend.floatx()) for i in experience["act_n"]]
        obs_next_n = [tf.convert_to_tensor(i, dtype=tf.keras.backend.floatx()) for i in experience["obs_next_n"]]

        # TODO INLINE NOISE(2)
        # noise_n = [[[
        #     tf.convert_to_tensor(i[:,j,k,:,:], dtype=tf.keras.backend.floatx())
        #     for k in range(i.shape[2])
        #     ] for j in range(i.shape[1])
        #     ] for i in experience["noise_n"]
        # ]
        # noise_n = [[[
        #     tf.convert_to_tensor(i[:,k,:,:], dtype=tf.keras.backend.floatx())
        #     for k in range(i.shape[1])
        #     ] for i in experience["noise_n"] for _ in range(4)]]
        noise_n_prim = [
            tf.convert_to_tensor(i, dtype=tf.keras.backend.floatx())
            for i in experience["noise_n"]]
        noise_n = [[noise_n_prim for _ in range(4)]]

        if self.args.integ_mode == '00':
            update_fn = self._update_00
        elif self.args.integ_mode == '01':
            update_fn = self._update_01
        elif self.args.integ_mode == '10':
            update_fn = self._update_10
        elif self.args.integ_mode == '11':
            update_fn = self._update_11
        else:
            update_fn = self._update_11

        q_loss, p_loss, target_q, target_q_next = update_fn(agents, experience, obs_n, act_n, obs_next_n, noise_n, mask)
        return np.array([q_loss, p_loss, np.mean(target_q), np.mean(experience["rew"]), np.mean(target_q_next), np.std(target_q)])

    @tf.function
    def _update_11(self, agents, experience, obs_n, act_n, obs_next_n, noise_n, mask):
        cct_obs, cct_act = tf.concat(obs_n, -1), tf.concat(act_n, -1)
        cct_obs_next = tf.concat(obs_next_n, -1)
        cct_ou_s = noise_n[0] #TODO MULTIPLE TRAINER SUPPORT

        ustck_obs = tf.squeeze(tf.concat(tf.unstack(obs_n, axis=-2), -1))
        ustck_obs_next = tf.squeeze(tf.concat(tf.unstack(obs_next_n, axis=-2), -1))
        ustck_act = tf.squeeze(tf.concat(tf.unstack(act_n, axis=-2), -1))

        self.polyak_rate_iterator += 1
        polyak_rate = 200
        num_sample = 1
        target_q = 0.0
        target_q_next = None

        for i in range(num_sample):
            target_act_next_a_n = self.target_actors.integ_sample(cct_obs_next, mask, cct_ou_s)
            target_q_next = self.target_critic.eval1(
                tf.concat(
                    (ustck_obs_next, tf.squeeze(tf.concat(tf.unstack(target_act_next_a_n, axis=-2), -1))),
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
            act, act_pd = self.actors.integ_sample_reg(cct_obs, mask, cct_ou_s)
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

    @tf.function
    def _update_00(self, agents, experience, obs_n, act_n, obs_next_n, noise_n, mask):
        cct_obs, cct_act = tf.concat(obs_n, -1), tf.concat(act_n, -1)
        cct_obs_next = tf.concat(obs_next_n, -1)
        cct_ou_s = noise_n[0]  # TODO MULTIPLE TRAINER SUPPORT

        ustck_obs = tf.squeeze(tf.concat(tf.unstack(obs_n, axis=-2), -1))
        ustck_obs_next = tf.squeeze(tf.concat(tf.unstack(obs_next_n, axis=-2), -1))
        ustck_act = tf.squeeze(tf.concat(tf.unstack(act_n, axis=-2), -1))

        self.polyak_rate_iterator += 1
        polyak_rate = 200
        num_sample = 1
        target_q = 0.0
        target_q_next = None

        for i in range(num_sample):
            target_act_next_a_n = self.target_actors.sample(cct_obs_next, mask)
            target_q_next = self.target_critic.eval1(
                tf.concat(
                    (ustck_obs_next, tf.squeeze(tf.concat(tf.unstack(target_act_next_a_n, axis=-2), -1))),
                    axis=-1
                )
            )
            target_q += experience["rew"] + self.args.gamma * (1.0 - experience["done"]) * tf.squeeze(target_q_next,
                                                                                                      axis=-1)
        target_q /= num_sample

        with tf.GradientTape() as tape_q:
            q = tf.squeeze(self.critic.eval2(tf.concat((ustck_obs, ustck_act), -1)), axis=-1)
            q_loss = tf.keras.losses.mean_squared_error(target_q, q)

        critic_grad = tape_q.gradient(q_loss, self.critic.trainable_weights)
        self.critic.optimizer.apply_gradients(zip(clipnorm(critic_grad, 0.5), self.critic.trainable_weights))

        with tf.GradientTape() as tape_p:
            act, act_pd = self.actors.sample_reg(cct_obs, mask)
            unstck_act_p = tf.squeeze(tf.concat(tf.unstack(act, axis=-2), -1))

            q = self.critic.eval(tf.concat((ustck_obs, unstck_act_p), -1))
            p_reg = tf.reduce_mean(tf.square(act_pd))
            p_loss = (-tf.reduce_mean(q) + p_reg * 1e-4) * 100

        actor_grad = tape_p.gradient(p_loss, self.actors.trainable_weights)
        # self.actors.optimizer.apply_gradients(zip(clipnorm(actor_grad, 0.5), self.actors.trainable_weights))
        self.actors.optimizer.apply_gradients(zip(actor_grad, self.actors.trainable_weights))

        if self.polyak_rate_iterator % polyak_rate == 0:
            update_target(self.actors, self.target_actors)
            update_target(self.critic, self.target_critic)

        return q_loss, p_loss, target_q, target_q_next

    @tf.function
    def _update_01(self, agents, experience, obs_n, act_n, obs_next_n, noise_n, mask):
        cct_obs, cct_act = tf.concat(obs_n, -1), tf.concat(act_n, -1)
        cct_obs_next = tf.concat(obs_next_n, -1)
        cct_ou_s = noise_n[0]  # TODO MULTIPLE TRAINER SUPPORT

        ustck_obs = tf.squeeze(tf.concat(tf.unstack(obs_n, axis=-2), -1))
        ustck_obs_next = tf.squeeze(tf.concat(tf.unstack(obs_next_n, axis=-2), -1))
        ustck_act = tf.squeeze(tf.concat(tf.unstack(act_n, axis=-2), -1))

        self.polyak_rate_iterator += 1
        polyak_rate = 200
        num_sample = 1
        target_q = 0.0
        target_q_next = None

        for i in range(num_sample):
            target_act_next_a_n = self.target_actors.sample(cct_obs_next, mask)
            target_q_next = self.target_critic.eval1(
                tf.concat(
                    (ustck_obs_next, tf.squeeze(tf.concat(tf.unstack(target_act_next_a_n, axis=-2), -1))),
                    axis=-1
                )
            )
            target_q += experience["rew"] + self.args.gamma * (1.0 - experience["done"]) * tf.squeeze(target_q_next,
                                                                                                      axis=-1)
        target_q /= num_sample

        with tf.GradientTape() as tape_q:
            q = tf.squeeze(self.critic.eval2(tf.concat((ustck_obs, ustck_act), -1)), axis=-1)
            q_loss = tf.keras.losses.mean_squared_error(target_q, q)

        critic_grad = tape_q.gradient(q_loss, self.critic.trainable_weights)
        self.critic.optimizer.apply_gradients(zip(clipnorm(critic_grad, 0.5), self.critic.trainable_weights))

        with tf.GradientTape() as tape_p:
            act, act_pd = self.actors.integ_sample_reg(cct_obs, mask, cct_ou_s)
            unstck_act_p = tf.squeeze(tf.concat(tf.unstack(act, axis=-2), -1))

            q = self.critic.eval(tf.concat((ustck_obs, unstck_act_p), -1))
            p_reg = tf.reduce_mean(tf.square(act_pd))
            p_loss = (-tf.reduce_mean(q) + p_reg * 1e-4) * 100

        actor_grad = tape_p.gradient(p_loss, self.actors.trainable_weights)
        # self.actors.optimizer.apply_gradients(zip(clipnorm(actor_grad, 0.5), self.actors.trainable_weights))
        self.actors.optimizer.apply_gradients(zip(actor_grad, self.actors.trainable_weights))

        if self.polyak_rate_iterator % polyak_rate == 0:
            update_target(self.actors, self.target_actors)
            update_target(self.critic, self.target_critic)

        return q_loss, p_loss, target_q, target_q_next

    @tf.function
    def _update_10(self, agents, experience, obs_n, act_n, obs_next_n, noise_n, mask):
        cct_obs, cct_act = tf.concat(obs_n, -1), tf.concat(act_n, -1)
        cct_obs_next = tf.concat(obs_next_n, -1)
        cct_ou_s = noise_n[0]  # TODO MULTIPLE TRAINER SUPPORT

        ustck_obs = tf.squeeze(tf.concat(tf.unstack(obs_n, axis=-2), -1))
        ustck_obs_next = tf.squeeze(tf.concat(tf.unstack(obs_next_n, axis=-2), -1))
        ustck_act = tf.squeeze(tf.concat(tf.unstack(act_n, axis=-2), -1))

        self.polyak_rate_iterator += 1
        polyak_rate = 200
        num_sample = 1
        target_q = 0.0
        target_q_next = None

        for i in range(num_sample):
            target_act_next_a_n = self.target_actors.integ_sample(cct_obs_next, mask, cct_ou_s)
            target_q_next = self.target_critic.eval1(
                tf.concat(
                    (ustck_obs_next, tf.squeeze(tf.concat(tf.unstack(target_act_next_a_n, axis=-2), -1))),
                    axis=-1
                )
            )
            target_q += experience["rew"] + self.args.gamma * (1.0 - experience["done"]) * tf.squeeze(target_q_next,
                                                                                                      axis=-1)
        target_q /= num_sample

        with tf.GradientTape() as tape_q:
            q = tf.squeeze(self.critic.eval2(tf.concat((ustck_obs, ustck_act), -1)), axis=-1)
            q_loss = tf.keras.losses.mean_squared_error(target_q, q)

        critic_grad = tape_q.gradient(q_loss, self.critic.trainable_weights)
        self.critic.optimizer.apply_gradients(zip(clipnorm(critic_grad, 0.5), self.critic.trainable_weights))

        with tf.GradientTape() as tape_p:
            act, act_pd = self.actors.sample_reg(cct_obs, mask)
            unstck_act_p = tf.squeeze(tf.concat(tf.unstack(act, axis=-2), -1))

            q = self.critic.eval(tf.concat((ustck_obs, unstck_act_p), -1))
            p_reg = tf.reduce_mean(tf.square(act_pd))
            p_loss = (-tf.reduce_mean(q) + p_reg * 1e-4) * 100

        actor_grad = tape_p.gradient(p_loss, self.actors.trainable_weights)
        # self.actors.optimizer.apply_gradients(zip(clipnorm(actor_grad, 0.5), self.actors.trainable_weights))
        self.actors.optimizer.apply_gradients(zip(actor_grad, self.actors.trainable_weights))

        if self.polyak_rate_iterator % polyak_rate == 0:
            update_target(self.actors, self.target_actors)
            update_target(self.critic, self.target_critic)

        return q_loss, p_loss, target_q, target_q_next
