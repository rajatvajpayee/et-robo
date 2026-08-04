class Optimizer:
    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError


class SGD(Optimizer):
    """
    Since, torch SGD expects every parameter to be nn.Parameter but 
    here CNN architecture is Manual. So, had to define SGD as well.
    """
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def step(self):
        self.model.step(self.lr)

    def zero_grad(self):
        self.model.zero_grad()