from .layers import (
    Conv2D,
    ReLU,
    Sigmoid,
    MaxPool2D,
    Flatten,
    Linear,
)

from .layers_fast import Conv2DFast, MaxPool2DFast

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

    def __init__(self, model_cfg):
        self.layers = []
        in_channels = model_cfg["input_channels"]
        num_classes = model_cfg["num_classes"]
        filters = model_cfg["cnn"]["filters"]
        kernel_size = model_cfg["cnn"]["kernel_size"]
        pool_size = model_cfg["cnn"]["pool_size"]
        hidden_dims = model_cfg["mlp"]["hidden_dims"]
        activation = model_cfg["activation"].lower()

        if activation == "relu":
            activation_layer = ReLU
        elif activation == "sigmoid":
            activation_layer = Sigmoid
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        image_size = 28

        # ---------------- CNN ---------------- #
        for out_channels in filters:
            self.layers.append(
                Conv2DFast(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                )
            )

            self.layers.append(activation_layer())

            self.layers.append(
                MaxPool2DFast(
                    kernel_size=pool_size,
                    stride=pool_size,
                )
            )

            in_channels = out_channels
            image_size //= pool_size

        # ---------------- Flatten ---------------- #
        self.layers.append(Flatten())

        in_features = in_channels * image_size * image_size

        # ---------------- MLP ---------------- #
        for hidden in hidden_dims:

            self.layers.append(
                Linear(
                    in_features=in_features,
                    out_features=hidden,
                )
            )

            self.layers.append(activation_layer())

            in_features = hidden

        # ---------------- Output ---------------- #
        self.layers.append(
            Linear(
                in_features=in_features,
                out_features=num_classes,
            )
        )
        
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
    

    def named_parameters(self):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "parameters"):
                params = layer.parameters()
                grads = layer.gradients()

                yield f"{layer.__class__.__name__}_{i}.weight", params[0], grads[0]
                yield f"{layer.__class__.__name__}_{i}.bias", params[1], grads[1]

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