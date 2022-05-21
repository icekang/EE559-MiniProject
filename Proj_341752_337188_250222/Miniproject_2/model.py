import torch
from torch import nn
from torch.nn.functional import fold, unfold


class Conv2d():
    def __init__(self, input_shape, kernel_size, depth, stride):
        # Batch size?
        input_depth, input_height, input_width = input_shape
        self.kernel_size = kernel_size
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.stride = stride
        out_size = lambda s_in, ks, st: int((( s_in - (ks - 1) - 1 ) / st + 1) // 1)
        self.output_shape = (depth, out_size(input_height, kernel_size, self.stride), out_size(input_width, kernel_size, self.stride))
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = torch.randn(self.kernels_shape)
        self.biases = torch.randn(depth)

    
    def set_initial(self, weight, bias):
        self.kernels = weight
        self.biases = bias
    
    
    def forward(self, input):
        input = input.type(torch.DoubleTensor)
        print(self.kernels_shape)
        self.input = input
        # change to empty
        self.output = torch.clone(self.biases)
        
        # do we need to add padding? 
        inp_unf = unfold(self.input.type(torch.DoubleTensor), kernel_size=self.kernels_shape[-2:], stride=2)

        out_unf = inp_unf.transpose(1, 2).matmul(self.kernels.view(inp_unf.size(1), -1).type(torch.DoubleTensor)).transpose(1, 2)

        self.output = fold(out_unf, output_size=self.output_shape[1:], kernel_size=(1,1), stride=1).permute(0,3,2,1)
        self.output += self.biases
        self.output = self.output.permute(0,3,2,1)

        return self.output
    
    def backward(self, output_gradient):
        ks = self.kernels_shape[-1] - 1
        self.output_gradient = output_gradient
        # dL/dK = X_j * output_grad
        inp_dldK = unfold(self.input, kernel_size=self.output_gradient.size()[-2:], stride = 2)
        print(self.output_gradient())
        out_dldK = inp_dldK.transpose(1, 2).matmul(self.output_gradient.view(self.output_gradient.size(1) * self.output_gradient.size(2), -1).t()).transpose(1, 2)
        
        
        # dL/db
        # dL/dX_j = sum_i dE/dYi * Kij
        
        
        self.kernel_flipped = self.kernels.flip([2,3])
        
        inp_unf = unfold(output_gradient, kernel_size=self.kernel_flipped[-2:], stride = 2, padding = (self.kernel_size - 1, self.kernel_size - 1))
        out_unf = inp_unf.transpose(1, 2).matmul(self.kernel_flipped.view(self.kernel_flipped.size(1) * self.kernel_flipped.size(2), -1).t()).transpose(1, 2)
        input_gradient = fold(out_unf, output_size=self.kernels_shape[1:], kernel_size=(1, 1), stride=2)

        # Maybe update the weights here??

        return input_gradient