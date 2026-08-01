import torch
import setproctitle
from tqdm import tqdm

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
            test_cfg["device"]
            if torch.cuda.is_available()
            else "cpu"
        )

        # Model
        self.model = SimpleCNN().to(self.device)

        # Load trained weights
        self.load_model(test_cfg["checkpoint"])

        # Data
        _, _, self.test_loader = get_dataloaders(
            batch_size=test_cfg["batch_size"]
        )

    def load_model(self, checkpoint):

        params = torch.load(checkpoint, map_location=self.device)

        idx = 0

        for layer in self.model.layers:

            if hasattr(layer, "weights"):
                layer.weights = params[idx].to(self.device)
                idx += 1

                layer.bias = params[idx].to(self.device)
                idx += 1

    def accuracy(self, logits, labels):

        predictions = torch.argmax(logits, dim=1)

        return (predictions == labels).float().mean().item()

    def test_step(self, x, y):

        logits = self.model.forward(x)

        loss = torch.nn.functional.cross_entropy(logits, y)

        acc = self.accuracy(logits, y)

        return loss.item(), acc

    def test(self):

        test_loss = 0.0
        test_acc = 0.0

        for x, y in tqdm(self.test_loader):

            x = x.to(self.device)
            y = y.to(self.device)

            loss, acc = self.test_step(x, y)

            test_loss += loss
            test_acc += acc

        test_loss /= len(self.test_loader)
        test_acc /= len(self.test_loader)

        print("\n========== Test Results ==========")
        print(f"Test Loss     : {test_loss:.4f}")
        print(f"Test Accuracy : {test_acc:.4f}")