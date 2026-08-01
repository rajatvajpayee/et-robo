from lzma import FILTER_LZMA1
import torch
import torch.nn as nn
import torch.nn.functional as F
import setproctitle

from src.data import get_dataloaders
from manual_nn.models import SimpleCNN
from src.utils import set_seed
from tqdm import tqdm 
from sklearn.metrics import f1_score

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import re

setproctitle.setproctitle("trainer.py")

class Trainer:
    def __init__(self, train_cfg, model_cfg, run_title):
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
        self.best_f1 = 0.0

        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_title != '':
            base_run_title = run_title
            runs_dir = "runs"
            # List all existing run directories that start with the base_run_title
            existing = [d for d in os.listdir(runs_dir) if d.startswith(base_run_title)]
            if existing:
                # Find runs of the form run_title, run_title_0, run_title_1, etc.
                pattern = re.compile(re.escape(base_run_title) + r"(?:_(\d+))?$")
                numbers = [-1]  # keep -1 for match with no trailing number (the plain name)
                for name in existing:
                    m = pattern.fullmatch(name)
                    if m:
                        if m.group(1) is not None:
                            numbers.append(int(m.group(1)))
                max_num = max(numbers)
                if max_num == -1:
                    run_title = f"{base_run_title}_1"
                else:
                    run_title = f"{base_run_title}_{max_num+1}"
            self.save_dir = f"runs/{run_title}"
        else:
            self.save_dir = f"runs/problem1_mnist_{current_time}"

        self.writer = SummaryWriter(self.save_dir)

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
        self.model.step(self.learning_rate) # optimizer directly added in the mode. may need to update
        acc = self.accuracy(logits, y)
        return loss.item(), acc

    def val_step(self, x, y):
        logits = self.model.forward(x)
        loss = self.loss_fn(logits, y)
        acc = self.accuracy(logits, y)

        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        targets = y.detach().cpu().numpy()
        f1 = f1_score(targets, preds, average='macro')
 
        return loss.item(), acc, f1

    def train(self):
        for epoch in range(self.epochs):
            train_loss = 0.0
            train_acc = 0.0

            val_loss = 0.0
            val_acc = 0.0
            val_f1 = 0.0

            for x, y in tqdm(self.train_loader, leave=False, desc=f"Training {len(self.train_loader)}"):
                x = x.to(self.device)
                y = y.to(self.device)
                loss, acc = self.train_step(x, y)
                train_loss += loss
                train_acc += acc
                self.writer.add_scalar("Loss/train", loss, epoch)
                self.writer.add_scalar("Accuracy/train", acc, epoch)
                self.writer.flush()

            train_loss /= len(self.train_loader)
            train_acc /= len(self.train_loader)
            

            # ---------------- Validation ---------------- #
            for x, y in tqdm(self.val_loader, leave = False, desc=f"Validation for {len(self.val_loader)}"):
                x = x.to(self.device)
                y = y.to(self.device)
                loss, acc, f1 = self.val_step(x, y)
                val_loss += loss
                val_acc += acc
                val_f1 += f1
                self.writer.add_scalar("Accuracy/val", acc, epoch)
                self.writer.add_scalar("F1/val", f1, epoch)
                self.writer.add_scalar("Loss/val", loss, epoch)
                self.writer.flush()

            val_loss /= len(self.val_loader)
            val_acc /= len(self.val_loader)
            val_f1 /= len(self.val_loader)

            self.train_loss.append(train_loss)
            self.train_accuracy.append(train_acc)

            self.val_loss.append(val_loss)
            self.val_accuracy.append(val_acc)
            if val_f1 >  self.best_f1:
                self.best_f1  = val_f1

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
                    f"{self.save_dir}/best_model.pth",
                )

            print(
                f"Epoch [{epoch + 1}/{self.epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        # Add TensorBoard writer support to log metrics
        # Log the experiment description and training config values
        experiment_desc = f"""
            ### Experiment: MNIST Baseline
            * **Model**: SimpleCNN 
            * **Optimizer**: {self.train_cfg.get('optimizer', 'N/A')}
            * **Learning Rate**: {self.train_cfg.get('learning_rate', 'N/A')}
            * **Loss**: {self.train_cfg.get('loss', 'N/A')}
            * **Epochs**: {self.train_cfg.get('epochs', 'N/A')}
            * **Batch Size**: {self.train_cfg.get('batch_size', 'N/A')}
            * **Seed**: {self.train_cfg.get('seed', 'N/A')}
            * **Device**: {self.train_cfg.get('device', 'N/A')}
            * **Notes**: Testing basic convergence on MNIST dataset.
        """

        self.writer.add_text('Experiment_Description', experiment_desc, global_step=0)

        # INSERT_YOUR_CODE
        # Prepare hyperparameters and metrics to log
        hparams = {
            'lr': self.train_cfg.get('learning_rate', 0.01),
            'bsize': self.train_cfg.get('batch_size', 64),
            'layers': len(getattr(self.model, 'layers', [])),
            'epochs': self.train_cfg.get('epochs', 20),
        }
        metrics = {
            'hparam/accuracy': self.best_val_accuracy,
            'hparam/loss': self.val_loss[-1] if self.val_loss else 0.0,
            'hparam/f1': self.best_f1,
        }
        self.writer.add_hparams(hparams, metrics)

        for name, weight, grad in self.model.named_parameters():
            self.writer.add_histogram(f"Weights/{name}", weight, epoch)
            self.writer.add_histogram(f"Gradients/{name}", grad, epoch)


        
        self.writer.close()
 

        print(f"\nBest Validation Accuracy: {self.best_val_accuracy:.4f}")