import argparse
from experiments.experiments.PubIntegBackground import PubIntegBackground

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--intensity", type=float, default=5.0, help="...")
    parser.add_argument("--repetitions", type=int, default=6, help="...")
    parser.add_argument("--integr", default='11', help="...")
    args, _ = parser.parse_known_args()

    for _ in range(args.repetitions):
        PubIntegBackground(correlation=False, listing=True, pub='None', integ_mode=args.integr, intensity=args.intensity, args=args)

