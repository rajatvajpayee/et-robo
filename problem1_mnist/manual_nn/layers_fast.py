"""Vectorised versions of Conv2D and MaxPool2D.

Same mathematics as layers.py, same manual forward/backward (no autograd anywhere)
— the only change is that the four nested Python loops are replaced by the
im2col identity:

    conv(x, W)  ==  W_flat @ unfold(x)          (a single matmul)

so the per-output-pixel loop happens inside one BLAS/cuDNN call instead of
inside the Python interpreter.

    forward:   out   = W_flat @ cols                     [B, Cout, L]
    backward:  dW    = sum_b  dout_b @ cols_b^T
               db    = dout.sum over (batch, H, W)
               dx    = fold( W_flat^T @ dout )

F.unfold / F.fold are pure tensor-reshaping utilities (im2col / col2im); they
contain no convolution logic, so the gradient derivation is still yours.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn.init as init

from .layers import Layer


class Conv2DFast(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weights = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        init.kaiming_normal_(self.weights, mode="fan_out")
        self.bias = torch.zeros(out_channels)

        self.grad_w = torch.zeros_like(self.weights)
        self.grad_b = torch.zeros_like(self.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        K, S, P = self.kernel_size, self.stride, self.padding
        self.x_shape = (B, C, H, W)

        # cols: [B, C*K*K, L] where L = H_out * W_out
        self.cols = F.unfold(x, kernel_size=K, padding=P, stride=S)
        H_out = (H + 2 * P - K) // S + 1
        W_out = (W + 2 * P - K) // S + 1
        self.out_hw = (H_out, W_out)

        w_flat = self.weights.reshape(self.out_channels, -1)      # [Cout, C*K*K]
        out = w_flat @ self.cols                                   # [B, Cout, L]
        out = out + self.bias.view(1, -1, 1)
        return out.reshape(B, self.out_channels, H_out, W_out)

    def backward(self, grad_output):
        B = grad_output.shape[0]
        K, S, P = self.kernel_size, self.stride, self.padding
        H, W = self.x_shape[2], self.x_shape[3]

        g = grad_output.reshape(B, self.out_channels, -1)          # [B, Cout, L]

        # dL/db : sum over batch and spatial positions
        self.grad_b = grad_output.sum(dim=(0, 2, 3))

        # dL/dW : contract the output-position axis with the patch axis
        self.grad_w = torch.einsum("bfl,bdl->fd", g, self.cols).reshape(self.weights.shape)

        # dL/dx : scatter back with col2im
        w_flat = self.weights.reshape(self.out_channels, -1)       # [Cout, C*K*K]
        dcols = w_flat.transpose(0, 1) @ g                         # [B, C*K*K, L]
        grad_input = F.fold(dcols, output_size=(H, W), kernel_size=K, padding=P, stride=S)
        return grad_input

    def step(self, lr):
        self.weights -= lr * self.grad_w
        self.bias -= lr * self.grad_b

    def zero_grad(self):
        self.grad_w.zero_()
        self.grad_b.zero_()

    def parameters(self):
        return [self.weights, self.bias]

    def gradients(self):
        return [self.grad_w, self.grad_b]

    def to(self, device):
        self.weights = self.weights.to(device)
        self.bias = self.bias.to(device)
        self.grad_w = self.grad_w.to(device)
        self.grad_b = self.grad_b.to(device)
        return self


class MaxPool2DFast(Layer):
    """Reshape-based max pool. Requires stride == kernel_size (the usual case)."""

    def __init__(self, kernel_size=2, stride=2):
        assert stride == kernel_size, "vectorised pool assumes non-overlapping windows"
        self.kernel_size = kernel_size
        self.stride = stride
        self.mask = None

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.kernel_size
        self.x_shape = (B, C, H, W)

        xr = x.reshape(B, C, H // k, k, W // k, k)
        out = xr.amax(dim=(3, 5))

        # one-hot mask over each window, ties broken toward the first element
        # (matches torch.argmax semantics)
        m = xr == out[:, :, :, None, :, None]
        m = m & (m.reshape(B, C, H // k, k, W // k, k)
                  .flatten(3, 3).cumsum(3).reshape(B, C, H // k, k, W // k, k)
                  .cumsum(5) == 1)
        self.mask = m
        return out

    def backward(self, grad_output):
        B, C, H, W = self.x_shape
        k = self.kernel_size
        grad_input = self.mask * grad_output[:, :, :, None, :, None]
        return grad_input.reshape(B, C, H, W)

    def to(self, device):
        return self