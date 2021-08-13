from experiments.commnet.commnet_experiment import CommnetExperiment
import maddpg.common.noise_fn as nfn


class ExperimentAbsoluteValue(CommnetExperiment):
    def __init__(self):
        super(ExperimentAbsoluteValue, self).__init__(name='absolute-commnet', args={'num_episodes': 60000})

        self.noise_r_fn = nfn.generate_noise(
            way=nfn.NoiseNames.WAY_MUL,
            type=nfn.NoiseNames.TYPE_PROBABILITY,
            val=nfn.NoiseNames.VALUE_CONSTANT,
            pck={
                'value': -1,
                'prob': 0.5
            }
        )
        self.noise_s_fn = nfn.identity
        self.trainers = self.get_trainers()

        shape = self.trainers[0].get_noise_shape()
        self.ou_manager = nfn.NoiseManagerOUNoCorrelation(shape)

        self.init()

