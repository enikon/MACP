from experiments.experiment import *
from experiments.mdmaddpg.mdmaddpg_trainer import MDMADDPGTrainer
from maddpg.common.replay_buffer import NReplayBuffer


class MDMADDPGExperiment(Experiment):
    def __init__(self):
        self.memory_state_in = None
        self.memory_init = None
        self.memory_a = None

        super(MDMADDPGExperiment, self).__init__(
            self.get_env,
            MDMADDPGTrainer,
            'md'
        )

        shape = self.trainers[0].actors.shape
        self.ou_manager = nfn.NoiseOUManager(
            [
                nfn.NoiseOU(shape, 0.2),
                nfn.NoiseUniform(shape),
                nfn.NoiseUniform(shape)
            ],
            [0, 0, 1, 2]
        )
        self.init()

    def parser(self):
        parser = super().parser()
        parser.add_argument("--disable-comm", action="store_true", default=False)
        parser.add_argument("--memory-size", type=int, default=200, help="size of the memory buffer for interagent communication")
        parser.add_argument("--encoder-units", type=int, default=512, help="---")
        parser.add_argument("--read-units", type=int, default=128, help="---")
        parser.add_argument("--action-units", type=int, default=256, help="---")
        return parser

    def init_loop(self):
        # Initialise memory
        self.memory_state_in = None
        self.memory_init = np.random.normal(loc=0.0, scale=1.0, size=(self.args.memory_size, )).astype(np.float32)

    def init_buffer(self):
        return NReplayBuffer(int(1e6), 6)

    def reset_loop(self):
        self.ou_manager.reset()
        self.memory_state_in = np.random.normal(loc=0.0, scale=1.0, size=(self.args.memory_size, )).astype(np.float32)

    def collect_action(self, obs_n, mask):
        # Populate actions, states for all agents
        action_n = []
        self.memory_a = []

        ou_s = self.ou_manager.get()
        mask = tf.constant([[1.], [1.], [0.]])

        for i, agent in enumerate(self.trainers):
            if self.args.disable_mem:
                self.memory_state_in = self.memory_state_in * 0
            act, mem = agent.action(obs_n[i][None], self.memory_state_in[None], mask[i], ou_s)

            action_n.append(act)
            self.memory_a.append(mem)
            self.memory_state_in = mem

        self.ou_manager.update()
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
    MDMADDPGExperiment()

