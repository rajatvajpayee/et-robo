import torch
import torch.nn.functional as F
import torch.nn.init as init

def conv2d(x, w, bias=None, stride=1, padding=0):
    """
    x: [B, C_in, H, W]
    w: [C_out, C_in, K, K]
    bias: [C_out] or None
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    B, C_in, H, W = x.shape
    C_out, C_in_w, K_h, K_w = w.shape
    assert C_in == C_in_w
    assert K_h == K_w, "Only square kernels supported here"

    x_pad = torch.nn.functional.pad(x, (padding[1], padding[1], padding[0], padding[0]))

    H_out = (x_pad.shape[2] - K_h) // stride[0] + 1
    W_out = (x_pad.shape[3] - K_w) // stride[1] + 1

    out = torch.zeros(B, C_out, H_out, W_out, device=x.device, dtype=x.dtype)

    for b in range(B):
        for oc in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride[0]
                    w_start = j * stride[1]
                    patch = x_pad[b, :, h_start:h_start + K_h, w_start:w_start + K_w]
                    out[b, oc, i, j] = (patch * w[oc]).sum()
                    if bias is not None:
                        out[b, oc, i, j] += bias[oc]

    return out


class SimpleCNN:
    def __init__(self, in_channels=1, out_channels=2,
                 kernel_size=3, num_classes=10, img_size=28):

        # Convolution parameters
        ### weights and biases for convolution layer
        # self.conv_w = torch.randn(
        #     out_channels, in_channels, kernel_size, kernel_size
        # ) * 0.01 # updating it to He/Kaiming initialization
        self.conv_w = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        init.kaiming_normal_(self.conv_w, mode="fan_out", nonlinearity="relu")
        self.conv_b = torch.zeros(out_channels)

        conv_out = img_size - kernel_size + 1
        flatten_dim = out_channels * conv_out * conv_out

        # Fully connected layer
        self.fc_w = torch.randn(flatten_dim, num_classes) * 0.01
        self.fc_b = torch.zeros(num_classes)

    def forward(self, x):
        """
        x : (B, C, H, W)
        """

        self.x = x

        # Manual convolution (using unfold)
        patches = conv2d(x, self.conv_w, self.conv_b, stride=1, padding=0)              # (B, 9, L)
        self.patches = patches

        w = self.conv_w.view(self.conv_w.shape[0], -1)     # (out_ch, 9)

        conv = torch.matmul(w, patches) + self.conv_b[:, None]
        # (B, out_ch, L)

        out_size = x.shape[-1] - 3 + 1

        conv = conv.view(
            x.shape[0],
            self.conv_w.shape[0],
            out_size,
            out_size
        )

        self.conv = conv

        # ReLU
        relu = torch.clamp(conv, min=0)
        self.relu = relu

        # Flatten
        flat = relu.reshape(x.shape[0], -1)
        self.flat = flat

        # FC
        logits = flat @ self.fc_w + self.fc_b
        self.logits = logits

        return logits

    def loss(self, y):
        """
        Cross-entropy loss (manual)
        """

        logits = self.logits

        exp = torch.exp(logits - logits.max(dim=1, keepdim=True)[0])
        probs = exp / exp.sum(dim=1, keepdim=True)

        self.probs = probs
        self.y = y

        loss = -torch.log(probs[torch.arange(len(y)), y]).mean()

        return loss.item()

    def backward(self):

        B = self.y.shape[0]

        # dL/dlogits
        grad_logits = self.probs.clone()
        grad_logits[torch.arange(B), self.y] -= 1
        grad_logits /= B

        # FC gradients
        grad_fc_w = self.flat.t() @ grad_logits
        grad_fc_b = grad_logits.sum(0)

        grad_flat = grad_logits @ self.fc_w.t()

        grad_relu = grad_flat.view_as(self.relu)

        # ReLU gradient
        grad_conv = grad_relu.clone()
        grad_conv[self.conv <= 0] = 0

        grad_conv = grad_conv.reshape(B, self.conv_w.shape[0], -1)

        # Conv weight gradients
        grad_conv_w = torch.zeros_like(self.conv_w)

        for oc in range(self.conv_w.shape[0]):
            g = grad_conv[:, oc]              # (B, L)

            for b in range(B):
                grad_conv_w[oc] += (
                    g[b][:, None] *
                    self.patches[b].t()
                ).sum(0).view_as(self.conv_w[oc])

        grad_conv_b = grad_conv.sum((0, 2))

        self.grad_fc_w = grad_fc_w
        self.grad_fc_b = grad_fc_b
        self.grad_conv_w = grad_conv_w
        self.grad_conv_b = grad_conv_b

    def step(self, lr=0.01):

        self.fc_w -= lr * self.grad_fc_w
        self.fc_b -= lr * self.grad_fc_b

        self.conv_w -= lr * self.grad_conv_w
        self.conv_b -= lr * self.grad_conv_b


# -----------------------------------------------------
# Example
# -----------------------------------------------------

model = SimpleCNN()

x = torch.randn(8, 1, 28, 28)
y = torch.randint(0, 10, (8,))

logits = model.forward(x)
loss = model.loss(y)

model.backward()
model.step(lr=0.01)

print("Loss:", loss)