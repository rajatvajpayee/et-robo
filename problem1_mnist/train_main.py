import setproctitle

from src.trainer import Trainer
from src.utils import load_config


setproctitle.setproctitle("training.py")

# INSERT_YOUR_CODE
import argparse

parser = argparse.ArgumentParser(description="Train manual NN model for MNIST.")

parser.add_argument(
    "--run_title",
    type=str,
    default="default",
    help="Tensorboard Run Title"
)

parser.add_argument(
    "--train_cfg",
    type=str,
    default="/home/rajat/scratch/et-robo/problem1_mnist/configs/training.yaml",
    help="Training Hyperparameters"
)

parser.add_argument(
    "--model_cfg",
    type=str,
    default="/home/rajat/scratch/et-robo/problem1_mnist/configs/model.yaml",
    help="Model Configuration"
)

args = parser.parse_args()



def main():

    train_cfg = load_config(args.train_cfg)
    model_cfg = load_config(args.model_cfg)


    trainer = Trainer(
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        run_title=args.run_title if args.run_title else ''
    )

    trainer.train()


if __name__ == "__main__":
    main()