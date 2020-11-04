import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description="This script plots training history",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input1", type=str, required=True,
                        help="path to input checkout directory 1 (must include history.npz)")
    parser.add_argument("--input2", type=str, default=None,
                        help="path to input checkout directory 2 (must include history.npz) "
                             "if you want to compare it with input1")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    input_paths = [Path(args.input1).joinpath("history.npz")]

    if args.input2:
        input_paths.append(Path(args.input2).joinpath("history.npz"))

    datum = [(np.array(np.load(str(input_path))["history"], ndmin=1)[0], input_path.parent.name)
             for input_path in input_paths]
    metrics = ["val_loss", "val_PSNR"]

    for metric in metrics:
        for data, setting_name in datum:
            plt.plot(data[metric], label=setting_name)
        plt.xlabel("epochs")
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(metric + ".png")
        plt.cla()


if __name__ == '__main__':
    main()
