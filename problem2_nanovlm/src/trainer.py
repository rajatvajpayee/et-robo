import ast
import torch
import torch.nn.functional as F
from tqdm import tqdm
import setproctitle
setproctitle.setproctitle("train.py")

class Trainer:

    def __init__(self, model, train_loader, val_loader, optimizer, device="cuda"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.best_acc = 0.0

    def train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch in tqdm(self.train_loader):
            self.optimizer.zero_grad()
            batch_loss = 0.0
            batch_correct = 0

            for sample in batch:
                processed_image, image_string = self.model.prepare_image(sample["image"])

                scores = []
                choices = ast.literal_eval(sample["choices"])

                for choice in choices:
                    input_ids, attention_mask = self.model.prepare_text(
                        image_string,
                        sample["question"],
                        choice,
                    )

                    score = self.model(
                        input_ids,
                        processed_image,
                        attention_mask,
                    )

                    scores.append(score.squeeze())

                scores = torch.stack(scores).unsqueeze(0)

                label = torch.tensor(
                    [sample["correct_answer"]],
                    dtype=torch.long,
                    device=self.device,
                )

                loss = F.cross_entropy(scores, label)
                loss.backward()

                batch_loss += loss.item()
                pred = scores.argmax(dim=1)
                batch_correct += (pred == label).sum().item()

            self.optimizer.step()

            total_loss += batch_loss
            correct += batch_correct
            total += len(batch)

        return total_loss / len(self.train_loader), correct / total

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for batch in tqdm(self.val_loader):
            for sample in batch:
                processed_image, image_string = self.model.prepare_image(sample["image"])

                scores = []
                choices = ast.literal_eval(sample["choices"])

                for choice in choices:
                    input_ids, attention_mask = self.model.prepare_text(
                        image_string,
                        sample["question"],
                        choice,
                    )

                    score = self.model(
                        input_ids,
                        processed_image,
                        attention_mask,
                    )

                    scores.append(score.squeeze())

                scores = torch.stack(scores).unsqueeze(0)

                label = torch.tensor(
                    [sample["correct_answer"]],
                    dtype=torch.long,
                    device=self.device,
                )

                loss = F.cross_entropy(scores, label)

                total_loss += loss.item()
                pred = scores.argmax(dim=1)
                correct += (pred == label).sum().item()
                total += 1

        return total_loss / total, correct / total

    def fit(self, epochs):
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss {train_loss:.4f} | "
                f"Train Acc {train_acc:.4f} | "
                f"Val Loss {val_loss:.4f} | "
                f"Val Acc {val_acc:.4f}"
            )

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.model.state_dict(), "best_model.pth")

if __name__ == "__main__":

    import torch

    from dataset import Robo2VLMDataModule
    from model import NanoVLMClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dm = Robo2VLMDataModule(
        train_path="/home/rajat/scratch/et-robo/data/Robo2VLM-1_local/train",
        test_path="/home/rajat/scratch/et-robo/data/Robo2VLM-1_local/test",
        batch_size=24,
        val_split=0.1,
        num_workers=0,
    )

    model = NanoVLMClassifier(device=device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-2,
    )

    trainer = Trainer(
        model=model,
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        optimizer=optimizer,
        device=device,
    )

    trainer.fit(epochs=1)

    