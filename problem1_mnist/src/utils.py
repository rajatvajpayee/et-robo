import yaml
import random
import torch
import numpy as np

def load_config(config_path="/home/rajat/scratch/et-robo/problem1_mnist/configs/data.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)