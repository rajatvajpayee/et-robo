import setproctitle

from src.tester import Tester
from src.utils import load_config

setproctitle.setproctitle("test_main.py")


def main():

    test_cfg = load_config(
        "/home/rajat/scratch/et-robo/problem1_mnist/configs/testing.yaml"
    )

    model_cfg = load_config(
        "/home/rajat/scratch/et-robo/problem1_mnist/configs/model.yaml"
    )

    tester = Tester(
        test_cfg=test_cfg,
        model_cfg=model_cfg,
    )

    tester.test()


if __name__ == "__main__":
    main()