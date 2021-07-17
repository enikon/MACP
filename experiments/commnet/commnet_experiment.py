from experiments.commnet.commnet_trainer import CommnetTrainer
from experiments.experiment import *
from maddpg.common.replay_buffer import NReplayBuffer


class CommnetExperiment(Experiment):
    def __init__(self):
        super(CommnetExperiment, self).__init__(
            self.get_env,
            CommnetTrainer,
            name='commnet'
        )

    def parser(self):
        parser = super().parser()
        parser.add_argument("--disable-comm", action="store_true", default=False)
        parser.add_argument("--communication_length", type=int, default=20, help="size of the communication vector")
        parser.add_argument("--communication_steps", type=int, default=2, help="number of communication messages sent")
        return parser

    def init_loop(self):
        pass

    def init_buffer(self):
        return NReplayBuffer(int(1e6), 6)

    def reset_loop(self):
        pass

    def collect_action(self, obs_n):
        # Populate actions, states for all agents
        action_n = []
        for i, agent in enumerate(self.trainers):
            act = agent.action(np.array(obs_n))
            action_n = act.numpy()
        return [np.squeeze(i) for i in np.split(action_n, action_n.shape[0])]

    def collect_experience(self, obs_n, action_n, rew_n, new_obs_n, done_n, terminal):
        for i, agent in enumerate(self.trainers):
            self.replay_buffer_n[i].add(obs_n, action_n, rew_n[0], new_obs_n, np.product(done_n), terminal)

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
                "agents",
                self.environment.n,
                self.environment.observation_space[0].shape,
                self.environment.action_space[0],
                self.args
            )
        ]


if __name__ == '__main__':
    CommnetExperiment()

