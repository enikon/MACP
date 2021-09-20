from experiments.experiments.ExperimentBaseline import *
from experiments.experiments.ExperimentBackground import *
from experiments.experiments.ExperimentAdaptingBackground import *
from experiments.experiments.ExperimentRetention import *
from experiments.experiments.ExperimentOther import *
from experiments.experiments.ExperimentBrokenAgent import *


if __name__ == "__main__":
    # for i in [0.1, 0.2, 0.3]:
    #     ExperimentRetention(correlation=False, probability=i)
    #ExperimentRetention(correlation=False, probability=0.4)

    #ExperimentRetention(correlation=True, probability=0.1)

    #for i in [0.1, 0.3]:
    #    ExperimentBackground(correlation=False, intensity=i)

    #
    #ExperimentAdaptingBackground(correlation=True, max_lim=0.3, intensity_range=(0.3, 0.3), speed=0.0)
    #ExperimentAdaptingBackground(correlation=True, max_lim=1.0, intensity_range=(1.0, 1.0), speed=0.0)

    #ExperimentBackground(correlation=True, intensity=0.3)

    #ExperimentAdaptingRetention(correlation=True, intensity_range=(-0.1, 0.5), max_lim=0.4, speed=0.6 / 60000.0)
    #ExperimentAdaptingRetention(correlation=True, intensity_range=(0.4, 0.4), max_lim=0.4, speed=0.0)
    #ExperimentAdaptingRetention(correlation=True, intensity_range=(0.0, 0.4), max_lim=0.2, speed=0.4 / 60000.0)
    ExperimentAdaptingRetention(correlation=True, intensity_range=(0.2, 0.2), max_lim=0.2, speed=0.0)
    #ExperimentRetention(correlation=True, probability=0.4)

    #ExperimentBackground(correlation=True, intensity=1.0)

    #ExperimentBrokenAgent(repMul=2, type='none')
