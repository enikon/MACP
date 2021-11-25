from experiments.commnet.commnet_trainer import CommnetTrainer
from experiments.experiment import *
from maddpg.common.replay_buffer import NReplayBuffer
import maddpg.common.noise_fn as nfn


class CommnetExperiment(Experiment):
    def __init__(self, trainer=CommnetTrainer, name='commnet', args=None):
        super(CommnetExperiment, self).__init__(
            self.get_env,
            trainer=trainer,
            name=name+(str(args['pub']) if args is not None else '')+str(args['integ_mode']),
            args=args
        )

    def init_default(self):
        self.noise_r_fn = nfn.identity
        self.noise_s_fn = nfn.identity

        self.trainers = self.get_trainers()

        shape = self.trainers[0].get_noise_shape()
        self.ou_manager = nfn.NoiseManagerOUNoCorrelation(shape)

        self.init()

    def parser(self):
        parser = super().parser()
        parser.add_argument("--disable-comm", action="store_true", default=False)
        parser.add_argument("--communication-length", type=int, default=256, help="size of the communication vector")
        parser.add_argument("--communication-steps", type=int, default=2, help="number of communication messages sent")
        parser.add_argument("--integ-mode", default='11', help="00, 01, 10, 11 first bit is on target_q second on policy")
        return parser

    def init_loop(self):
        pass

    def init_buffer(self):
        return NReplayBuffer(int(5e5), 7)

    def reset_loop(self):
        self.ou_manager.reset()

    def collect_action(self, obs_n, mask):
        # Populate actions, states for all agents
        action_n = []
        self.ou_manager.update()
        ou_s = self.ou_manager.get()

        # if self.noise_adapting is not None:
        #     ou_s = self.expand_adapting(ou_s)

        for i, agent in enumerate(self.trainers):
            act = agent.action(np.array(obs_n), mask, ou_s)
            action_n = act.numpy()
        return [np.squeeze(i) for i in np.split(action_n, action_n.shape[0])]

    def collect_metrics(self, obs_n, mask):
        ou_s = self.ou_manager.get()
        metrics = [None for _ in range(4)]

        # if self.noise_adapting is not None:
        #     ou_s = self.expand_adapting(ou_s)

        for agent in self.trainers: #there is only one trainer in commnet, can be ignored
            m = agent.actors.metrics_call(
                (np.expand_dims(obs_n, 0), mask, ou_s))
            for i, m_ in enumerate(m):
                metrics[i] = tf.stack(m_)

        stacked_metrics = tf.stack([tf.stack(metrics[:2], 0), tf.stack(metrics[2:], 0)], 0)
        return tf.math.reduce_euclidean_norm(stacked_metrics, (2, 3, 5))
        #tf.math.reduce_variance(stacked_metrics, (-1, 2))

    def collect_experience(self, obs_n, action_n, rew_n, new_obs_n, done_n, terminal, mask):
        for i, agent in enumerate(self.trainers):
            self.replay_buffer_n[i].add(obs_n, action_n, rew_n[0], new_obs_n, np.product(done_n), terminal, mask)

    def train_experience(self, buffer):

        index = buffer.make_index(self.args.batch_size)

        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []

        for b in self.replay_buffer_n:
            obs, act, _, obs_next, _, _, _ = b.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done, _, mask = buffer.sample_index(index)

        return {
            "obs_n": obs_n,
            "obs_next_n": obs_next_n,
            "act_n": act_n,
            "obs": obs,
            "act": act,
            "rew": rew,
            "obs_next": obs_next,
            "done": done,
            "mask": mask
        }

    def get_trainers(self):
        return [
            self.trainer(
                "agents",
                self.environment.n,
                self.environment.observation_space[0].shape,
                self.environment.action_space[0],
                self.args,
                noise_r_fn=self.noise_r_fn,
                noise_s_fn=self.noise_s_fn,
                noise_adapting=self.noise_adapting,
                pub=self.args.pub
            )
        ]

    # def expand_adapting(self, ou_s):
    #     return (
    #         (ou_s[0], self.noise_adapting_value) if self.noise_adapting[0] else ou_s[0],
    #         (ou_s[1], self.noise_adapting_value) if self.noise_adapting[1] else ou_s[1],
    #         (ou_s[2], self.noise_adapting_value) if self.noise_adapting[2] else ou_s[2],
    #         (ou_s[3], self.noise_adapting_value) if self.noise_adapting[3] else ou_s[3]
    #     )


if __name__ == '__main__':
    ce = CommnetExperiment()
    ce.init_default()

