import setproctitle

from src.tester import Tester
from src.utils import load_config

setproctitle.setproctitle("testing.py")

import argparse

parser = argparse.ArgumentParser(description="Test a trained MNIST model.")

parser.add_argument(
    "--test_cfg",
    type=str,
    default="/home/rajat/scratch/et-robo/problem1_mnist/configs/testing.yaml",
    help="Path to testing config YAML.",
)
parser.add_argument(
    "--model_cfg",
    type=str,
    default="/home/rajat/scratch/et-robo/problem1_mnist/configs/model.yaml",
    help="Path to model config YAML.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to model checkpoint.",
)


args = parser.parse_args()

def main():
    tester = Tester(args)
    tester.test()

if __name__ == "__main__":
    main()