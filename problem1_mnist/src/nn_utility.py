class Layer:
    """
    Base class for all neural network layers.
    """

    def __init__(self):
        pass

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Output tensor.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement forward()."
        )

    def backward(self, grad_output):
        """
        Backward pass.

        Args:
            grad_output: Gradient of the loss with respect
                         to this layer's output.

        Returns:
            Gradient of the loss with respect
            to this layer's input.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement backward()."
        )

    def step(self, lr):
        """
        Update learnable parameters using gradients.

        Layers without learnable parameters
        (e.g., ReLU, Flatten) do not need to override this.
        """
        pass

    def zero_grad(self):
        """
        Reset gradients before the next backward pass.

        Parameterized layers should override this.
        """
        pass

class ReLU(Layer):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask
