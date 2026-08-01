import torch
import torch.nn as nn
import torch.nn.functional as F
import setproctitle

from src.data import get_dataloaders
from manual_nn.models import SimpleCNN
from src.utils import set_seed
from tqdm import tqdm 

setproctitle.setproctitle("trainer.py")


class Trainer:
    def __init__(self, train_cfg, model_cfg):
        self.train_cfg = train_cfg
        self.model_cfg = model_cfg

        set_seed(train_cfg["seed"])

        self.device = torch.device(
            train_cfg["device"]
            if torch.cuda.is_available()
            else "cpu"
        )
        print(self.device)
        # Model
        self.model = SimpleCNN().to(self.device)

        # Data
        self.train_loader, self.val_loader, _ = get_dataloaders(
            batch_size=train_cfg["batch_size"]
        )
        print("Loaded All Dataloaders")

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        self.epochs = train_cfg["epochs"]
        self.learning_rate = train_cfg["learning_rate"]

        # History
        self.train_loss = []
        self.train_accuracy = []

        self.val_loss = []
        self.val_accuracy = []

        self.best_val_accuracy = 0.0

    def accuracy(self, logits, labels):
        predictions = torch.argmax(logits, dim=1)
        return (predictions == labels).float().mean().item()

    def train_step(self, x, y):

        # Forward
        logits = self.model.forward(x)

        # Loss (PyTorch)
        loss = self.loss_fn(logits, y)

        # Manual gradient of CE
        probs = F.softmax(logits, dim=1)

        targets = F.one_hot(
            y,
            num_classes=self.model_cfg["num_classes"]
        ).float()

        grad_logits = (probs - targets) / y.size(0)

        # Manual backward
        self.model.zero_grad()
        self.model.backward(grad_logits)
        self.model.step(self.learning_rate)

        acc = self.accuracy(logits, y)

        return loss.item(), acc

    def val_step(self, x, y):
        logits = self.model.forward(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)
        return loss.item(), acc

    def train(self):
        for epoch in range(self.epochs):
            train_loss = 0.0
            train_acc = 0.0

            val_loss = 0.0
            val_acc = 0.0

            for x, y in tqdm(self.train_loader, leave=False, desc=f"Training {len(self.train_loader)}"):
                x = x.to(self.device)
                y = y.to(self.device)
                loss, acc = self.train_step(x, y)
                train_loss += loss
                train_acc += acc

            train_loss /= len(self.train_loader)
            train_acc /= len(self.train_loader)
            

            # ---------------- Validation ---------------- #
            for x, y in tqdm(self.val_loader, leave = False, desc=f"Validation for {len(self.val_loader)}"):
                x = x.to(self.device)
                y = y.to(self.device)
                loss, acc = self.val_step(x, y)
                val_loss += loss
                val_acc += acc

            val_loss /= len(self.val_loader)
            val_acc /= len(self.val_loader)

            self.train_loss.append(train_loss)
            self.train_accuracy.append(train_acc)

            self.val_loss.append(val_loss)
            self.val_accuracy.append(val_acc)

            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                checkpoint = {}
                layer_id = 0
                for layer in self.model.layers:
                    if hasattr(layer, "weights"):
                        checkpoint[f"layer_{layer_id}.weights"] = layer.weights.clone()
                        checkpoint[f"layer_{layer_id}.bias"] = layer.bias.clone()
                        layer_id += 1
                torch.save(
                    checkpoint,
                    "outputs/best_model.pth",
                )
                print("Best model saved.")

            print(
                f"Epoch [{epoch + 1}/{self.epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        print(f"\nBest Validation Accuracy: {self.best_val_accuracy:.4f}")