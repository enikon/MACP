from experiments.experiments.ExperimentBaseline import *
from experiments.experiments.ExperimentBackground import *
from experiments.experiments.ExperimentSpreadBaseline import *
from experiments.experiments.ExperimentRetention import *
from experiments.experiments.ExperimentOther import *
from experiments.experiments.ExperimentBrokenAgent import *


if __name__ == "__main__":
    # for i in [0.1, 0.2, 0.3]:
    #     ExperimentRetention(correlation=False, probability=i)
    # ExperimentRetention(correlation=True, probability=0.2)

    #for i in [0.1, 0.3]:
    #    ExperimentBackground(correlation=False, intensity=i)

    ExperimentNoCommunication()
    #ExperimentDisabledCommunication()


    #ExperimentBrokenAgent(repMul=0, type='fixed')
