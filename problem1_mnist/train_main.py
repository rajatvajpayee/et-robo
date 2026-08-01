import setproctitle

from src.trainer import Trainer
from src.utils import load_config


setproctitle.setproctitle("train_main.py")


def main():

    train_cfg = load_config("/home/rajat/scratch/et-robo/problem1_mnist/configs/training.yaml")
    model_cfg = load_config("/home/rajat/scratch/et-robo/problem1_mnist/configs/model.yaml")


    trainer = Trainer(
        train_cfg=train_cfg,
        model_cfg=model_cfg,
    )

    trainer.train()


if __name__ == "__main__":
    main()