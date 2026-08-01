from .layers import (
    Conv2D,
    ReLU,
    MaxPool2D,
    Flatten,
    Linear,
)


class Model:
    """Base Model (Factory)"""
    def forward(self, x):
        raise NotImplementedError
    def backward(self, grad):
        raise NotImplementedError
    def step(self, lr):
        raise NotImplementedError
    def zero_grad(self):
        raise NotImplementedError


class SimpleCNN(Model):
    """
    Architecture

    Input (1x28x28)
          │
    Conv2D(1 -> 8, 3x3)
          │
        ReLU
          │
      MaxPool2D
          │
      Flatten
          │
    Linear(1352 -> 10)
    """

    def __init__(self):

        self.layers = [
            Conv2D(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            Flatten(),
            Linear(8 * 14 * 14, 10),
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, grad):
        for layer in reversed(self.layers): # reversed because gradients starts from the loss and ends at input
            grad = layer.backward(grad)
        return grad

    def step(self, lr):
        for layer in self.layers:
            layer.step(lr)

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, "parameters"):
                params.extend(layer.parameters())
        return params

    def gradients(self):
        grads = []
        for layer in self.layers:
            if hasattr(layer, "gradients"):
                grads.extend(layer.gradients())
        return grads

    def __repr__(self):
        model = "SimpleCNN(\n"
        for layer in self.layers:
            model += f"    {layer}\n"
        model += ")"
        return model
    
    def to(self, device):
        for layer in self.layers:
            if hasattr(layer, "to"):
                layer.to(device)
        return self

if __name__ == "__main__":
    """Execute : python -m manual_nn.models to test it separately as it has relative import of layers"""

    import torch

    model = SimpleCNN()

    print(model)

    x = torch.randn(4, 1, 28, 28)

    logits = model.forward(x)

    print("\nOutput Shape:", logits.shape)

    grad = torch.randn_like(logits)

    model.backward(grad)

    model.step(0.001)

    print("\nForward and Backward Successful!")