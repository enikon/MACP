from experiments.experiment import *
from experiments.maddpg.maddpg_trainer import MADDPGTrainer
from maddpg.common.replay_buffer import NReplayBuffer


class MADDPGExperiment(Experiment):
    def __init__(self):
        super(MADDPGExperiment, self).__init__(
            self.get_env,
            MADDPGTrainer,
            'maddpg'
        )

    def init_loop(self):
        pass

    def init_buffer(self):
        return NReplayBuffer(int(1e6), 6)

    def reset_loop(self):
        pass

    def collect_action(self, obs_n):
        return [agent.action(obs).numpy() for agent, obs in zip(self.trainers, obs_n)]

    def collect_experience(self, obs_n, action_n, rew_n, new_obs_n, done_n, terminal):
        for i, agent in enumerate(self.trainers):
            self.replay_buffer_n[i].add(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)

    def train_experience(self, buffer):

        index = buffer.make_index(self.args.batch_size)

        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []

        for b in self.replay_buffer_n:
            obs, act, _, obs_next, _, _ = b.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done, _ = buffer.sample_index(index)

        return {
            "obs_n": obs_n,
            "obs_next_n": obs_next_n,
            "act_n": act_n,
            "obs": obs,
            "act": act,
            "rew": rew,
            "obs_next": obs_next,
            "done": done
        }

    def get_trainers(self):
        return [
            self.trainer(
                "agent_%d" % i,
                self.environment.n,
                self.environment.observation_space[i].shape,
                self.environment.action_space[i],
                i,
                self.args
            )
            for i in range(self.environment.n)
        ]


if __name__ == '__main__':
    MADDPGExperiment()
