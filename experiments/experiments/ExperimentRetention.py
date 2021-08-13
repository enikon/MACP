from experiments.commnet.commnet_experiment import CommnetExperiment
import maddpg.common.noise_fn as nfn


class ExperimentRetention(CommnetExperiment):
    def __init__(self, correlation=False, probability=0.1):
        super(ExperimentRetention, self).__init__(
            name='retention-'+('corr-' if correlation else 'nocorr-')+(str(probability))+'-commnet', args={'num_episodes': 60000})

        self.noise_r_fn = nfn.generate_noise(
            way=nfn.NoiseNames.WAY_REP,
            type=nfn.NoiseNames.TYPE_PROBABILITY,
            val=nfn.NoiseNames.VALUE_UNIFORM,
            pck={
                'range': (-1.0, 1.0),
                'prob': probability
            }
        )

        self.noise_s_fn = nfn.identity
        self.trainers = self.get_trainers()

        shape = self.trainers[0].get_noise_shape()

        if not correlation:
            self.ou_manager = nfn.NoiseManagerOUNoCorrelation(shape)
        else:
            self.ou_manager = nfn.NoiseOUManager(
                [
                    nfn.NoiseUniform(shape),  # sgi, svi, rvi
                    nfn.NoiseOU(shape, 0.02, (0, 1))  # rgi
                ],
                [0, 0, 1, 0]
            )
        self.init()

