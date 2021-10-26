from experiments.experiments.PubIntegBackground import PubIntegBackground
import numpy as np

if __name__ == "__main__":
    for i in np.arange(0.0, 10.0, 0.1):
        PubIntegBackground(correlation=False, listing=True, pub='None', intensity=i)
