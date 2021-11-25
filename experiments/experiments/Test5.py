import argparse
from experiments.experiments.PubIntegBackground import PubIntegBackground

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--intensity", type=float, default=5.0, help="...")
    parser.add_argument("--repetitions", type=int, default=6, help="...")
    args, _ = parser.parse_known_args()

    for _ in range(args.repetitions):
        PubIntegBackground(correlation=False, listing=True, pub='None', integ_mode='00', intensity=args.intensity)
        PubIntegBackground(correlation=False, listing=True, pub='None', integ_mode='01', intensity=args.intensity)
        PubIntegBackground(correlation=False, listing=True, pub='None', integ_mode='10', intensity=args.intensity)
        PubIntegBackground(correlation=False, listing=True, pub='None', integ_mode='11', intensity=args.intensity)

