import torch
import torch.nn.init as init
import torch.nn.functional as F



def conv2d(x, weight, bias=None, stride=1, padding=0):
    """
    Y = Conv(X, W) + b 
    """

    B, C_in, H, W = x.shape
    C_out, _, K, _ = weight.shape

    # Pad input
    x = F.pad(x, (padding, padding, padding, padding))

    H_out = (H + 2 * padding - K) // stride + 1
    W_out = (W + 2 * padding - K) // stride + 1

    out = torch.zeros(B, C_out, H_out, W_out,
                        device=x.device, dtype=x.dtype)

    for b in range(B):
        for c in range(C_out):
            for h in range(H_out):
                for w in range(W_out):

                    h_start = h * stride
                    w_start = w * stride

                    patch = x[b,:,h_start:h_start + K,w_start:w_start + K]

                    out[b, c, h, w] = (patch * weight[c]).sum()

                    if bias is not None:
                        out[b, c, h, w] += bias[c]

    return out

class Layer:
    """Base class for NN layers."""
    def __init__(self):
        pass
    def forward(self, x):
        raise NotImplementedError
    def backward(self, grad_output):
        raise NotImplementedError
    def step(self, lr):
        pass
    def zero_grad(self):
        pass
    def to(self,device):
        pass

class Linear(Layer):
    """
    FULLY CONNECTED LAYER
    Y = XW + b
    where, W is weights, b is bias, X is input, Y is output
    Dimensions:
    X: (batch_size, in_features)
    W: (out_features, in_features)
    b: (out_features)
    Y: (batch_size, out_features)
    """
    def __init__(self, in_features, out_features):
        # Parameters
        self.weights = torch.empty(in_features, out_features)
        init.kaiming_normal_(self.weights, mode='fan_out') 
        # init.xavier_uniform_(self.weights)
        self.bias = torch.zeros(out_features) 

        # Gradients
        self.grad_w = torch.zeros_like(self.weights)
        self.grad_b = torch.zeros_like(self.bias)
        
    
    def forward(self, x):
        self.x = x 
        return x @ self.weights + self.bias
    
    def backward(self, grad_output):
        """
        backward pass for linear layer
        grad_output = dL/dY, where L is loss function and Y is output of linear layer
        dL/dW = dL/dY * dY/dW 
        dL/db = dL/dY * dY/db 
        dL/dx = dL/dY * dY/dx 
        """
        #dL/dW 
        self.grad_w = self.x.T @ grad_output   
        #dL/db
        self.grad_b = grad_output.sum(dim=0)
        #dL/dx
        grad_input = grad_output @ self.weights.T
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
        

class ReLU(Layer):
    """ReLU Activation"""
    def __init__(self):
        self.mask = None 
    
    def forward(self, x):
        """
               /
              /        
        _____/
        mask is required because any grad value less than 0 should be set to 0
        """
        self.mask = x > 0 
        return x * self.mask 

    def backward(self, grad_output):
        return grad_output * self.mask
    def to(self, device):
        return self

class Sigmoid(Layer):
    """Sigmoid Activation Layer"""
    def __init__(self):
        self.output = None
    def forward(self, x):
        # Numerically stable sigmoid
        self.output = 1 / (1 + torch.exp(-x))
        return self.output
    def backward(self, grad_output):
        # d(sigmoid)/dx = s * (1 - s)
        return grad_output * self.output * (1 - self.output)
    def to(self, device):
        return self



class Flatten(Layer):
    """Flatten Layer"""
    def __init__(self):
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(self.input_shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)
    
    def to(self, device):
        return self


class Conv2D(Layer):
    """2D Convolutional Layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights and bias
        self.weights = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        init.kaiming_normal_(self.weights, mode='fan_out')
        # init.xavier_uniform_(self.weights)
        self.bias = torch.zeros(out_channels)

        # Gradients
        self.grad_w = torch.zeros_like(self.weights)
        self.grad_b = torch.zeros_like(self.bias)


    def forward(self, x):
        self.x = x
        return conv2d(x, self.weights, self.bias, stride=self.stride, padding=self.padding)

    def backward(self, grad_output):
        """
        for one channel : Y_c = Conv(...) + b_c
        dL/dW_c = dL/dY_c * dY_c/dW_c : (kernel_size, kernel_size); grad value for each element in the kernel
        dL/db_c = dL/dY_c * dY_c/db_c : (1, )
        dL/dx_c = dL/dY_c * dY_c/dx_c : (Height, Width)


        dL/dW : 
        input : 
        1 2 3
        4 5 6
        7 8 9 

        kernel : 
        a b
        c d


        forward pass : a*1 + b*2 + c*4 + d*5 
        Assume loss value for this channel is g.
        params : a,b,c,d 

        So, dL/da = g, dL/db = 2g, dL/dc = 4g, dL/dd = 5g
        dL/dW = g + 2g + 4g + 5g = g * [1,2,4,5]

        """
        #dL/db_c
        # input size is (Batch, Channels, Height, Width)
        self.grad_b = grad_output.sum(dim=(0,2,3)) # sum over batch, height, width. so, output size is (out_channels, )

        #dL/dW_c
        B, _, H_out, W_out = grad_output.shape
        # Padded input (same as forward)
        if self.padding > 0:
            x = torch.nn.functional.pad(
                self.x,
                (self.padding, self.padding, # left, right
                self.padding, self.padding)  # top, bottom
            )
        else:
            x = self.x

        for b in range(B):
            for c in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride

                        patch = x[b, :, h_start:h_start+self.kernel_size, w_start:w_start+self.kernel_size]
                        self.grad_w[c] += (patch * grad_output[b, c, h, w])

        # dL/dx
        if self.padding > 0:
            x = F.pad(self.x, (self.padding,) * 4)
            grad_input = torch.zeros_like(x)
        else:
            x = self.x
            grad_input = torch.zeros_like(x)

        for b in range(B):
            for c in range(self.out_channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        grad_input[b,:,h_start:h_start + self.kernel_size,w_start:w_start + self.kernel_size,] += (
                            self.weights[c] * grad_output[b, c, h, w]
                        )

        if self.padding > 0:
            """
            Remove padding from grad_input if padding is used.
            """
            grad_input = grad_input[:,:,self.padding:-self.padding,self.padding:-self.padding]

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
    

class MaxPool2D(Layer):
    """Max Pooling Layer"""

    def __init__(self, kernel_size=2, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride
        self.mask = None

    def forward(self, x):
        self.x = x
        B, C, H, W = x.shape
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        out = torch.zeros(B, C, H_out, W_out, device=x.device,dtype=x.dtype)

        # Stores the position of the maximum values
        self.mask = torch.zeros_like(x)

        for b in range(B):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        patch = x[b,c,h_start:h_start+self.kernel_size,w_start:w_start+self.kernel_size]

                        max_value = patch.max()

                        out[b, c, h, w] = max_value

                        # Position of max inside the patch
                        max_idx = patch.argmax()

                        max_h = max_idx // self.kernel_size
                        max_w = max_idx % self.kernel_size

                        self.mask[b,c,h_start + max_h,w_start + max_w] = 1

        return out

    def backward(self, grad_output):
        """
        Does max pooling has parameters? No. Reason is, it doesn't have any weights to update. 
        But we need dL/dx to backpropagate the gradient. 
        dL/dx = dL/dY * dY/dx 
        Y = Conv(X, W) + b 
        dY/dx = W
        dL/dx = dL/dY * W
        """

        grad_input = torch.zeros_like(self.x)
        B, C, H_out, W_out = grad_output.shape
        for b in range(B):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        w_start = w * self.stride

                        patch_mask = self.mask[
                            b,
                            c,
                            h_start:h_start+self.kernel_size,
                            w_start:w_start+self.kernel_size,
                        ]

                        grad_input[
                            b,
                            c,
                            h_start:h_start+self.kernel_size,
                            w_start:w_start+self.kernel_size,
                        ] += (
                            patch_mask * grad_output[b, c, h, w]
                        )

        return grad_input

if __name__ == "__main__":

    """
    Test cases for the layers. (Used LLM here fully to write the driver code to test every class in layer)
    """
    torch.manual_seed(42)

    # --------------------------------------------------
    # Dummy input
    # --------------------------------------------------
    x = torch.randn(10, 1, 28, 28)

    print("=" * 50)
    print("Input")
    print(x.shape)

    # --------------------------------------------------
    # Conv Layer
    # --------------------------------------------------
    conv = Conv2D(
        in_channels=1,
        out_channels=4,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    conv_out = conv.forward(x)

    print("\nConv Output")
    print(conv_out.shape)

    grad = torch.randn_like(conv_out)

    grad_conv = conv.backward(grad)

    print("Conv Backward")
    print(grad_conv.shape)

    # --------------------------------------------------
    # ReLU
    # --------------------------------------------------
    relu = ReLU()

    relu_out = relu.forward(conv_out)

    print("\nReLU Output")
    print(relu_out.shape)

    grad_relu = relu.backward(torch.randn_like(relu_out))

    print("ReLU Backward")
    print(grad_relu.shape)

    # --------------------------------------------------
    # MaxPool
    # --------------------------------------------------
    pool = MaxPool2D(kernel_size=2, stride=2)

    pool_out = pool.forward(relu_out)

    print("\nMaxPool Output")
    print(pool_out.shape)

    grad_pool = pool.backward(torch.randn_like(pool_out))

    print("MaxPool Backward")
    print(grad_pool.shape)

    # --------------------------------------------------
    # Flatten
    # --------------------------------------------------
    flatten = Flatten()

    flat = flatten.forward(pool_out)

    print("\nFlatten Output")
    print(flat.shape)

    grad_flat = flatten.backward(torch.randn_like(flat))

    print("Flatten Backward")
    print(grad_flat.shape)

    # --------------------------------------------------
    # Linear
    # --------------------------------------------------
    linear = Linear(flat.shape[1], 10)

    logits = linear.forward(flat)

    print("\nLinear Output")
    print(logits.shape)

    grad_linear = linear.backward(torch.randn_like(logits))

    print("Linear Backward")
    print(grad_linear.shape)

    # --------------------------------------------------
    # SGD Step
    # --------------------------------------------------
    linear.step(0.01)
    conv.step(0.01)

    print("\nParameter update successful!")

    print("=" * 50)