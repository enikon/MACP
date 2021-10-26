from experiments.commnet.commnet_experiment import CommnetExperiment
import maddpg.common.noise_fn as nfn


class ExperimentBrokenAgent(CommnetExperiment):
    def __init__(self, repMul=0, type='fixed'):
        super(ExperimentBrokenAgent, self).__init__(name='broken-'+'rep-'if repMul==0 else 'mul-'+type+'-commnet', args={'num_episodes': 60000, 'noise_mask': type, 'noise_mask_value': 3, 'scenario': 'simple_reference_4'})

        if repMul == 0:
            self.noise_s_fn = nfn.generate_noise(
                way=nfn.NoiseNames.WAY_REP,
                type=nfn.NoiseNames.TYPE_ALL,
                val=nfn.NoiseNames.VALUE_UNIFORM,
                pck={
                    'range': (-1.0, 1.0)
                }
            )
        elif repMul == 1:
            self.noise_s_fn = nfn.generate_noise(
                way=nfn.NoiseNames.WAY_MUL,
                type=nfn.NoiseNames.TYPE_ALL,
                val=nfn.NoiseNames.VALUE_UNIFORM,
                pck={
                    'range': (-5.0, 5.0)
                }
            )
        elif repMul == 2:
            self.noise_s_fn = nfn.generate_noise(
                way=nfn.NoiseNames.WAY_MUL,
                type=nfn.NoiseNames.TYPE_ALL,
                val=nfn.NoiseNames.VALUE_CONSTANT,
                pck={
                    'value': 1.0
                }
            )


        self.noise_r_fn = nfn.identity
        self.trainers = self.get_trainers()

        shape = self.trainers[0].get_noise_shape()
        self.ou_manager = nfn.NoiseManagerOUNoCorrelation(shape)

        self.init()

