from experiments.experiment import *
from maddpg.common.replay_buffer import NReplayBuffer


class CommnetExperiment(Experiment):
    def __init__(self):
        self.memory_state_in = None
        self.memory_init = None
        self.memory_a = None

        super(CommnetExperiment, self).__init__(
            self.get_env,
            CommnetExperiment,
            name='commnet'
        )

    def parser(self):
        parser = super().parser()
        parser.add_argument("--disable-comm", action="store_true", default=False)
        parser.add_argument("--communication_length", type=int, default=20, help="size of the communication vector")
        parser.add_argument("--communication_steps", type=int, default=3, help="number of communication messages sent")
        return parser

    def init_loop(self):
        pass

    def init_buffer(self):
        return NReplayBuffer(int(1e6), 7)

    def reset_loop(self):
        pass

    def collect_action(self, obs_n):
        # Populate actions, states for all agents
        action_n = []
        self.memory_a = []

        for i, agent in enumerate(self.trainers):
            if self.args.disable_mem:
                self.memory_state_in = self.memory_state_in * 0

            act, mem = agent.action(obs_n[i][None], self.memory_state_in[None])

            action_n.append(act)
            self.memory_a.append(mem)
            self.memory_state_in = mem
        return action_n

    def collect_experience(self, obs_n, action_n, rew_n, new_obs_n, done_n, terminal):
        for i, agent in enumerate(self.trainers):
            self.replay_buffer_n[i].add(obs_n[i], self.memory_a[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)

    def train_experience(self, buffer):

        index = buffer.make_index(self.args.batch_size)

        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        memory_n = []

        for b in self.replay_buffer_n:
            obs, mem, act, _, obs_next, _, _ = b.sample_index(index)
            obs_n.append(obs)
            memory_n.append(mem)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, mem, act, rew, obs_next, done, _ = buffer.sample_index(index)

        return {
            "obs_n": obs_n,
            "memory_n": memory_n,
            "obs_next_n": obs_next_n,
            "act_n": act_n,
            "obs": obs,
            "mem": mem,
            "act": act,
            "rew": rew,
            "obs_next": obs_next,
            "done": done
        }


if __name__ == '__main__':
    CommnetExperiment()

