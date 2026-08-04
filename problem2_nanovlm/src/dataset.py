import numpy as np
from datasets import load_from_disk
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class Robo2VLMDataModule:

    def __init__(
        self,
        train_path,
        test_path,
        batch_size=8,
        val_split=0.1,
        num_workers=4,
        seed=42,
    ):
        train_dataset = load_from_disk(train_path)
        test_dataset = load_from_disk(test_path)

        idx = list(range(len(train_dataset)))
        labels = train_dataset["correct_answer"]

        train_idx, val_idx = train_test_split(
            idx,
            test_size=0.1,
            random_state=42,
            stratify=labels,
        )

        self.train_dataset = train_dataset.select(train_idx)
        self.val_dataset = train_dataset.select(val_idx)
        self.test_dataset = test_dataset

        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=lambda x: x,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda x: x,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=lambda x: x,
        )

def main(train_path, test_path, batch_size=8, num_workers=2):
    # Assuming the class above is named Something like 'YourDataClass'
    # Replace 'YourDataClass' with the actual class name if needed
    # For demonstration will assume the outer class is named "Robo2VLMDataModule"
    # If it's not, replace accordingly.
    obj = Robo2VLMDataModule(
        train_path=train_path,
        test_path=test_path,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    train_loader = obj.train_dataloader()
    val_loader = obj.val_dataloader()
    test_loader = obj.test_dataloader()
    print(f"train loader size: {len(train_loader)}")
    print(f"val loader size: {len(val_loader)}")
    print(f"test loader size: {len(test_loader)}")
    # Dataset is a dictionary. SO, it returns a list of 8 items. 
    print("train batch shape:", len(next(iter(train_loader))))
    print("val batch shape:", len(next(iter(val_loader))))
    print("test batch shape:", len(next(iter(test_loader))))

if __name__ == "__main__":
    train_path = '/home/rajat/scratch/et-robo/data/Robo2VLM-1_local/train'
    test_path = '/home/rajat/scratch/et-robo/data/Robo2VLM-1_local/test'
    batch_size = 8
    num_workers = 4
    main(train_path, test_path, batch_size, num_workers)
 