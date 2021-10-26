from experiments.commnet.commnet_experiment import CommnetExperiment
import maddpg.common.noise_fn as nfn


class PubBackground(CommnetExperiment):
    def __init__(self, correlation=False, listing=True, intensity=1.0, pub=None):
        super(PubBackground, self).__init__(name='pub-background-'+str(intensity)+'-'+('corr-' if correlation else 'nocorr-')+(str(intensity))+'-commnet', args={'num_episodes': 60000, 'pub':pub})
        self.noise_r_fn = nfn.generate_noise(
            way=nfn.NoiseNames.WAY_ADD,
            type=nfn.NoiseNames.TYPE_ALL,
            val=nfn.NoiseNames.VALUE_UNIFORM,
            pck={
                'range': (-intensity, intensity)
            }
        )
        self.noise_s_fn = nfn.identity

        self.trainers = self.get_trainers()
        if listing:
            shape = [self.trainers[0].get_noise_shape()]*self.trainers[0].actors.c_layers
        else:
            shape = [self.trainers[0].get_noise_shape()]

        if not correlation:
            self.ou_manager = nfn.NoiseManagerOUNoCorrelation(shape)
        else:
            self.ou_manager = nfn.NoiseOUManager(
                [
                    nfn.NoiseUniform(shape),  # sgi, rgi
                    nfn.NoiseOU(shape, 0.02, (-1, 1))  # rvi
                ],
                [0, 0, 0, 1]
            )

        self.init()
