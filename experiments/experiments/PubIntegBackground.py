import maddpg.common.noise_fn as nfn
from experiments.commnet.integ_commnet_experiment import IntegCommnetExperiment


class PubIntegBackground(IntegCommnetExperiment):
    def __init__(self, correlation=False, listing=True, intensity=1.0, pub=None, integ_mode='11', args=None):
        _args = {'num_episodes': 60000, 'pub': pub, 'integ_mode': integ_mode}
        if args is not None:
            _args.update(args.__dict__)

        super(PubIntegBackground, self).__init__(name='pub-integ-Nbackground-'+str(intensity)+'-'+('corr-' if correlation else 'nocorr-')+(str(intensity))+'-commnet', args=_args)

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
        for i in self.trainers:
            i.addOU([j.simulate for j in self.ou_manager.noise_get])

        self.init()
