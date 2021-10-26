from experiments.experiments.PubBackground import PubBackground
import numpy as np
if __name__ == "__main__":
    for i in np.arange(0.0, 10.0, 0.1):
        PubBackground(correlation=False, listing=True, pub=None, intensity=i)
