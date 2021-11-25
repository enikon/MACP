from experiments.experiments.PubIntegBackground import PubIntegBackground
from os import listdir
from os.path import join
import argparse


if __name__ == "__main__":

    exp_list = [(0, 0.), (1, 1.), (2, 2.5), (3, 5.), (4, 10.), (5, 25.), (6, 50.), (7, 100.), (8, 200.)]
    parser = argparse.ArgumentParser("")
    parser.add_argument("--group-load-dir", default='/tmp/tmp/load', help="...")
    parser.add_argument("--group-save-dir", default='/tmp/tmp/save', help="...")
    args, _ = parser.parse_known_args()

    groups_names = listdir(args.group_load_dir)

    for i, g in enumerate(groups_names):
        group_load_dir = join(args.group_load_dir, g)
        group_save_dir = join(args.group_save_dir, g)
        for j, gg in enumerate(listdir(group_load_dir)):
            lp = join(group_load_dir, gg)
            sp = join(group_save_dir, gg)

            for _, i in exp_list:
                PubIntegBackground(correlation=False, listing=True, pub=None, intensity=i,
                                          args={'save_dir': sp, 'load_dir': lp})

# (0,0); (1,1); (2, 2.5), (3, 5), (4, 10), (5,25), (6,50) (7, 100), (8, 200)
