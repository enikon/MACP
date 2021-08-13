from experiments.Trainer import Trainer
import numpy as np
import tensorflow as tf
from maddpg.model.maddpg import Actor, Critic
from maddpg.common.tf_util import clipnorm, update_target
from maddpg.model.mdmaddpg import MDActorNetwork
import maddpg.common.noise_fn as nfn


class MDMADDPGTrainer(Trainer):
    def __init__(self, name, n, obs_shape, act_space, agent_index, args):

        self.name = name
        self.n = n
        self.agent_index = agent_index
        self.args = args

        self.polyak_rate_iterator = 0

        self.actor = MDActorNetwork(act_space=act_space, args=args)
        self.target_actor = MDActorNetwork(act_space=act_space, args=args)

        self.critic = Critic(args=args)
        self.target_critic = Critic(args=args)

        self.noise_s_fn = nfn.identity
        self.noise_r_fn = nfn.generate_noise(
            shape=(1, self.args.memory_size),
            way=nfn.NoiseNames.WAY_ADD,
            type=nfn.NoiseNames.TYPE_PROBABILITY,
            val=nfn.NoiseNames.VALUE_CONSTANT,
            pck={
                'value': 2,
                'prob': 0.2
            })

    def model_save_dict(self):
        return {
            "critic": self.critic,
            "target_critic": self.target_critic,
            "actor": self.actor,
            "target_actor": self.target_actor
        }

    @tf.function
    def action(self, obs, mem, mask, ou_a):

        read_mem_with_noise = mem #self.noise_r_fn(mem, mask, (ou_s[0], ou_s[1]))
        p, m, ou_a_res = self.actor.sample_ou(obs, read_mem_with_noise, ou_a)
        write_mem_with_noise = m #self.noise_s_fn(m, mask, (ou_s[0], ou_s[1]))

        return p[0], write_mem_with_noise[0], ou_a_res

    def update(self, agents, experience):
        obs_n = [tf.convert_to_tensor(i, dtype=tf.keras.backend.floatx()) for i in experience["obs_n"]]
        memory_n = [tf.convert_to_tensor(i, dtype=tf.keras.backend.floatx()) for i in experience["memory_n"]]
        act_n = [tf.convert_to_tensor(i, dtype=tf.keras.backend.floatx()) for i in experience["act_n"]]
        obs_next_n = [tf.convert_to_tensor(i, dtype=tf.keras.backend.floatx()) for i in experience["obs_next_n"]]

        q_loss, p_loss, target_q, target_q_next = self._update(agents, experience, obs_n, memory_n, act_n, obs_next_n)
        return np.array(
            [q_loss, p_loss, np.mean(target_q), np.mean(experience["rew"]), np.mean(target_q_next), np.std(target_q)])

    @tf.function
    def _update(self, agents, experience, obs_n, memory_n, act_n, obs_next_n):
        cct_obs, cct_act = tf.concat(obs_n, -1), tf.concat(act_n, -1)
        cct_obs_next = tf.concat(obs_next_n, -1)

        self.polyak_rate_iterator += 1
        polyak_rate = 5
        num_sample = 1
        target_q = 0.0
        target_q_next = None

        for i in range(num_sample):
            target_act_next_a_n = [agents[i].target_actor.sample(obs_next_n[i], memory_n[i])[0] for i in range(self.n)]
            target_q_next = self.target_critic.eval1(
                tf.concat((cct_obs_next, tf.concat(target_act_next_a_n, -1)), -1))
            target_q += experience["rew"] + self.args.gamma * (1.0 - experience["done"]) * tf.squeeze(target_q_next, axis=-1)
        target_q /= num_sample

        with tf.GradientTape() as tape_q:
            q = tf.squeeze(self.critic.eval2(tf.concat((cct_obs, cct_act), -1)), axis=-1)
            q_loss = tf.keras.losses.mean_squared_error(target_q, q)

        critic_grad = tape_q.gradient(q_loss, self.critic.trainable_weights)
        self.critic.optimizer.apply_gradients(zip(clipnorm(critic_grad, 0.5), self.critic.trainable_weights))

        with tf.GradientTape() as tape_p:
            act_a, _, act_pd = self.actor.sample_reg(experience["obs"], experience["mem"])
            act_n_new = [a for a in act_n]
            act_n_new[self.agent_index] = act_a
            q = self.critic.eval(tf.concat((cct_obs, tf.concat(act_n_new, -1)), -1))

            p_reg = tf.reduce_mean(tf.square(act_pd))
            p_loss = (-tf.reduce_mean(q) + p_reg * 1e-4) * 100
        actor_grad = tape_p.gradient(p_loss, self.actor.trainable_weights)
        self.actor.optimizer.apply_gradients(zip(clipnorm(actor_grad, 0.5), self.actor.trainable_weights))

        if self.polyak_rate_iterator % polyak_rate == 0:
            update_target(self.actor, self.target_actor)
            update_target(self.critic, self.target_critic)

        return q_loss, p_loss, target_q, target_q_next

    def get_networks(self):
        return [self.actor, self.target_actor, self.critic, self.target_critic]