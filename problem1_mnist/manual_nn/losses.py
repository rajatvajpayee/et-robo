import torch

class Loss:
    """Base class for all losses"""
    def __init__(self):
        raise NotImplementedError

    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        raise NotImplementedError

class CrossEntropyLoss(Loss):
    def __init__(self):
        self.logits = None
        self.targets = None
        self.probs = None

    def forward(self, logits, targets):
        self.logits = logits
        self.targets = targets
        self.probs = torch.softmax(logits, dim=1)
        loss = -torch.sum(
            targets * torch.log(self.probs)
        ) / logits.shape[0]

        grad = (self.probs - self.targets) / self.logits.shape[0]

        return loss, grad


if __name__ == "__main__":

    torch.manual_seed(42)

    # -----------------------------
    # Dummy data
    # -----------------------------
    batch_size = 10
    num_classes = 10

    logits = torch.randn(batch_size, num_classes)

    labels = torch.randint(0, num_classes, (batch_size,))

    # Convert labels to one-hot
    targets = torch.nn.functional.one_hot(
        labels,
        num_classes=num_classes
    ).float()

    # -----------------------------
    # Manual Cross Entropy
    # -----------------------------
    criterion = CrossEntropyLoss()

    loss,grad = criterion.forward(logits, targets)


    print("Loss:", loss.item())
    print("Gradient Shape:", grad.shape)

    # -----------------------------
    # Compare with PyTorch
    # -----------------------------
    torch_loss = torch.nn.CrossEntropyLoss()

    loss_torch = torch_loss(logits, labels)

    print("\nPyTorch Loss :", loss_torch.item())
    print("Manual Loss  :", loss.item())

    print("\nDifference :", abs(loss_torch.item() - loss.item()))