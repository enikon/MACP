from experiments.commnet.commnet_experiment import CommnetExperiment
from experiments.commnet.integ_commnet_trainer import IntegCommnetTrainer
from experiments.experiment import *
from maddpg.common.replay_buffer import NReplayBuffer


class IntegCommnetExperiment(CommnetExperiment):
    def __init__(self, name='integ-commnet', args=None):
        super(IntegCommnetExperiment, self).__init__(
            trainer=IntegCommnetTrainer, name=name, args=args
        )

    def init_default(self):
        self.trainers = self.get_trainers()
        for i in self.trainers:
            i.addOU([j.simulate for j in self.ou_manager.noise_get])
        super().init_default()

    def init_buffer(self):
        return NReplayBuffer(int(5e5), 10)

    def collect_action(self, obs_n, mask):
        # Populate actions, states for all agents
        action_n = []
        ou_s = self.ou_manager.get()

        # if self.noise_adapting is not None:
        #     ou_s = self.expand_adapting(ou_s)

        for i, agent in enumerate(self.trainers):
            act = agent.action(np.array(obs_n), mask, ou_s)
            action_n = act.numpy()
        return [np.squeeze(i) for i in np.split(action_n, action_n.shape[0])]

    def collect_experience(self, obs_n, action_n, rew_n, new_obs_n, done_n, terminal, mask):

        pre_ous = tf.squeeze(self.ou_manager.pre_get())[3]
        ous = tf.squeeze(self.ou_manager.get())[3]
        self.ou_manager.update()
        ous_next = tf.squeeze(self.ou_manager.get())[3]

        for i, agent in enumerate(self.trainers):
            #TODO INLINE NOISE(1)
            #ous = [tf.squeeze(j) for j in self.ou_manager.get()]
            self.replay_buffer_n[i].add(obs_n, action_n, rew_n[0], new_obs_n, np.product(done_n), terminal, mask, ous, ous_next, pre_ous)

    def no_collect_experience(self):
        self.ou_manager.update()

    def train_experience(self, buffer):

        index = buffer.make_index(self.args.batch_size)

        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        noise_n = []
        noise_next_n = []
        pre_noise_n = []

        for b in self.replay_buffer_n:
            obs, act, _, obs_next, _, _, _, noise, noise_next, pre_noise = b.sample_index(index)

            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
            noise_n.append(noise)
            noise_next_n.append(noise_next)
            pre_noise_n.append(pre_noise)

        obs, act, rew, obs_next, done, _, mask, noise, noise_next, pre_noise = buffer.sample_index(index)

        return {
            "obs_n": obs_n,
            "obs_next_n": obs_next_n,
            "act_n": act_n,
            "obs": obs,
            "act": act,
            "rew": rew,
            "obs_next": obs_next,
            "done": done,
            "mask": mask,
            "noise_n": noise_n,
            "noise": noise,
            "noise_n_next": noise_next_n,
            "noise_next": noise_next,
            "pre_noise_n": pre_noise_n,
            "pre_noise": pre_noise
        }


if __name__ == '__main__':
    ce = IntegCommnetExperiment()
    ce.init_default()

