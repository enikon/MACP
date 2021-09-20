from experiments.experiments.ExperimentBaseline import *
from experiments.experiments.ExperimentBackground import *
from experiments.experiments.ExperimentAdaptingBackground import *
from experiments.experiments.ExperimentRetention import *
from experiments.experiments.ExperimentOther import *
from experiments.experiments.ExperimentBrokenAgent import *


if __name__ == "__main__":

    #ExperimentRetention(correlation=True, probability=0.1)

    #ExperimentRetention(correlation=True, probability=0.2)
    #ExperimentRetention(correlation=False, probability=0.5)

    #ExperimentAbsoluteValue()

    #for i in [0.1, 0.3]:
    #    ExperimentBackground(correlation=False, intensity=i)
    #ExperimentAdaptingBackground(correlation=True, intensity_range=(-0.3, 1.3), speed=1.6/60000.0)

    ExperimentAdaptingRetention(correlation=True, intensity_range=(-0.1, 0.5), max_lim=0.4, speed=0.6 / 60000.0)

    #ExperimentBrokenAgent(repMul=1, type='fixed')
