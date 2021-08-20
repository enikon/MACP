from experiments.commnet.commnet_experiment import CommnetExperiment
import maddpg.common.noise_fn as nfn
import tensorflow as tf


class ExperimentAdaptingBackground(CommnetExperiment):
    def __init__(self, correlation=False, intensity_range=(0.0, 1.0), speed=1.0/60000.0):
        super(ExperimentAdaptingBackground, self).__init__(name='adapt_background-'+('corr-' if correlation else 'nocorr-')+(str(speed))+'-'+(str(intensity_range[0]))+'-'+(str(intensity_range[1]))+'-commnet', args={'num_episodes': 60000})

        self.noise_r_fn = nfn.generate_noise(
            way=nfn.NoiseNames.WAY_ADD,
            type=nfn.NoiseNames.TYPE_ALL,
            val=nfn.NoiseNames.VALUE_VARIABLE
        )
        self.noise_s_fn = nfn.generate_noise(
            way=nfn.NoiseNames.WAY_ADD,
            type=nfn.NoiseNames.TYPE_ALL,
            val=nfn.NoiseNames.VALUE_VARIABLE
        )

        self.noise_adapting = [False, True, False, True]
        self.noise_adapting_value_iterator = tf.constant(intensity_range[0], dtype=tf.float32)
        self.noise_adapting_speed = tf.constant(speed, dtype=tf.float32)
        self.noise_adapting_max = tf.constant(intensity_range[1], dtype=tf.float32)

        self.trainers = self.get_trainers()
        shape = self.trainers[0].get_noise_shape()

        if not correlation:
            self.ou_manager = nfn.NoiseManagerOUNoCorrelation(shape)
        else:
            self.ou_manager = nfn.NoiseOUManager(
                [
                    nfn.NoiseUniform(shape),  # sgi, rgi
                    nfn.NoiseOU(shape, 0.02, (-1, 1)),  # svi
                    nfn.NoiseOU(shape, 0.02, (-1, 1))  # rvi
                ],
                [0, 1, 0, 2]
            )

        self.init()
