from experiments.experiments.ExperimentBaseline import *
from experiments.experiments.ExperimentBackground import *
from experiments.experiments.ExperimentAdaptingBackground import *
from experiments.experiments.ExperimentRetention import *
from experiments.experiments.ExperimentOther import *
from experiments.experiments.ExperimentBrokenAgent import *


if __name__ == "__main__":
    for i in [0.1, 0.3, 0.4, 0.5]:
         ExperimentRetention(correlation=True, probability=i)
    #ExperimentRetention(correlation=False, probability=0.4)
    #ExperimentRetention(correlation=False, probability=0.5)

    #for i in [0.1, 0.3]:
    #    ExperimentBackground(correlation=False, intensity=i)
    #ExperimentAdaptingBackground(correlation=True, intensity_range=(-0.3, 1.3), speed=1.6/60000.0)

    #ExperimentBrokenAgent(repMul=0, type='fixed')
