from experiments.experiments.ExperimentBaseline import *
from experiments.experiments.ExperimentBackground import *
from experiments.experiments.ExperimentRetention import *
from experiments.experiments.ExperimentOther import *
from experiments.experiments.ExperimentBrokenAgent import *


if __name__ == "__main__":
    ExperimentFullCommunication()
    ExperimentFullNoise()
    for i in [0.5, 1.0, 2.0]:
        ExperimentBackground(correlation=False, intensity=i)
    ExperimentBackground(correlation=True, intensity=1.0)

    for i in [0.1, 0.2, 0.3]:
        ExperimentRetention(correlation=False, probability=i)
    ExperimentRetention(correlation=True, probability=0.2)

    ExperimentAbsoluteValue()

    ExperimentBrokenAgent(type='fixed')
    ExperimentBrokenAgent(type='random')
