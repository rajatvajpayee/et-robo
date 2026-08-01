import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.utils import load_config

def get_dataloaders(data_dir="/home/rajat/scratch/et-robo/data/mnist_local", batch_size=64, num_workers=4):

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=transform
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    config = load_config(config_path=os.path.join("/home/rajat/scratch/et-robo/problem1_mnist/configs/data.yaml"))
    data_dir = config["root"]
    batch_size = config["batch_size"]
    num_workers = config["num_workers"]

    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size, num_workers)
    print(f"train_dataset size : {len(train_loader  )}")
    print(f"  val_dataset size : {len(val_loader    )}")
    print(f" test_dataset size : {len(test_loader   )}")