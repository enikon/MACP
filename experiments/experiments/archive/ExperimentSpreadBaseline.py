from experiments.commnet.commnet_experiment import CommnetExperiment
import maddpg.common.noise_fn as nfn


class ExperimentFullCommunication(CommnetExperiment):
    def __init__(self):
        super(ExperimentFullCommunication, self).__init__(name='spread-full-comm-commnet', args={'num_episodes': 60000, 'scenario': 'simple_reference_spread'})

        self.noise_r_fn = nfn.identity
        self.noise_s_fn = nfn.identity

        self.trainers = self.get_trainers()

        shape = self.trainers[0].get_noise_shape()
        self.ou_manager = nfn.NoiseManagerOUNoCorrelation(shape)
        self.init()


class ExperimentNoCommunication(CommnetExperiment):
    def __init__(self):
        super(ExperimentNoCommunication, self).__init__(name='spread-0-comm-commnet', args={'num_episodes': 60000, 'scenario': 'simple_reference_spread'})

        self.noise_r_fn = nfn.generate_noise(
            way=nfn.NoiseNames.WAY_REP,
            type=nfn.NoiseNames.TYPE_ALL,
            val=nfn.NoiseNames.VALUE_CONSTANT,
            pck={
                'value': 0.0
            }
        )
        self.noise_s_fn = nfn.identity
        self.trainers = self.get_trainers()

        shape = self.trainers[0].get_noise_shape()
        self.ou_manager = nfn.NoiseManagerOUNoCorrelation(shape)
        self.init()


class ExperimentDisabledCommunication(CommnetExperiment):
    def __init__(self):
        super(ExperimentDisabledCommunication, self).__init__(name='spread-dis-comm-commnet', args={'num_episodes': 60000, 'scenario': 'simple_reference_spread', 'disable_comm': True})

        self.noise_r_fn = nfn.identity
        self.noise_s_fn = nfn.identity

        self.trainers = self.get_trainers()

        shape = self.trainers[0].get_noise_shape()
        self.ou_manager = nfn.NoiseManagerOUNoCorrelation(shape)
        self.init()


class ExperimentFullNoise(CommnetExperiment):
    def __init__(self):
        super(ExperimentFullNoise, self).__init__(name='spread-no-comm-commnet', args={'num_episodes': 60000, 'scenario': 'simple_reference_spread'})

        self.noise_r_fn = nfn.generate_noise(
            way=nfn.NoiseNames.WAY_REP,
            type=nfn.NoiseNames.TYPE_ALL,
            val=nfn.NoiseNames.VALUE_UNIFORM,
            pck={
                'range': (-1.0, 1.0)
            }
        )
        self.noise_s_fn = nfn.identity
        self.trainers = self.get_trainers()

        shape = self.trainers[0].get_noise_shape()
        self.ou_manager = nfn.NoiseManagerOUNoCorrelation(shape)
        self.init()

