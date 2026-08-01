# Problems Faced 

1. Since, it is not optmized. I am using pythin loops which are slow. Hence, slower computation as compared to standard torch library. 
2. Convolution implementation: Handling stride, padding, output dimensions, and multi-channel convolutions correctly.
3. MaxPool backward: Tracking the max index during the forward pass and routing gradients only to those locations.
4. Device management: Keeping all tensors, parameters, and intermediate outputs on the same device (CPU/GPU).
5. Performance: The naive nested-loop implementation is significantly slower than PyTorch's optimized C++/CUDA kernels.
6. Performance: I/O overhead (CPU - total time 11 hrs while GPU - 20hrs)