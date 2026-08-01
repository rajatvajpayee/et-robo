import torch
import torch.nn.functional as F
import setproctitle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data import get_dataloaders
from manual_nn.models import SimpleCNN
from src.utils import set_seed

setproctitle.setproctitle("tester.py")


class Tester:
    def __init__(self, test_cfg, model_cfg):
        self.test_cfg = test_cfg
        self.model_cfg = model_cfg
        set_seed(test_cfg["seed"])

        self.device = torch.device(
            test_cfg["device"] if torch.cuda.is_available() else "cpu"
        )

        self.model = SimpleCNN().to(self.device)
        self.load_model(test_cfg["checkpoint"])

        _, _, self.test_loader = get_dataloaders(
            batch_size=test_cfg["batch_size"]
        )

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        layer_id = 0
        for layer in self.model.layers:
            if hasattr(layer, "weights"):
                layer.weights = checkpoint[f"layer_{layer_id}.weights"].to(self.device)
                layer.bias = checkpoint[f"layer_{layer_id}.bias"].to(self.device)
                layer_id += 1

        print(f"Loaded model from {checkpoint_path}")

    def eval_metrics(self, logits, labels):
        probs = F.softmax(logits, dim=1).cpu().numpy()
        labels = labels.cpu().numpy()
        preds = np.argmax(probs, axis=1)

        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average="weighted", zero_division=0),
            "recall": recall_score(labels, preds, average="weighted", zero_division=0),
            "f1": f1_score(labels, preds, average="weighted", zero_division=0),
        }

        try:
            metrics["auc"] = roc_auc_score(
                np.eye(probs.shape[1])[labels],
                probs,
                multi_class="ovr",
                average="macro",
            )
        except Exception:
            metrics["auc"] = float("nan")

        return metrics

    def test_step(self, x, y):
        logits = self.model.forward(x)
        loss = F.cross_entropy(logits, y)
        return loss.item(), logits, y

    def test(self):
        test_loss = 0.0
        all_logits = []
        all_labels = []

        for x, y in tqdm(self.test_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            loss, logits, labels = self.test_step(x, y)

            test_loss += loss
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

        test_loss /= len(self.test_loader)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metrics = self.eval_metrics(all_logits, all_labels)

        print("\n========== Test Results ==========")
        print(f"Loss      : {test_loss:.4f}")
        print(f"Accuracy  : {metrics['accuracy']:.4f}")
        print(f"Precision : {metrics['precision']:.4f}")
        print(f"Recall    : {metrics['recall']:.4f}")
        print(f"F1 Score  : {metrics['f1']:.4f}")
        print(f"ROC AUC   : {metrics['auc']:.4f}")