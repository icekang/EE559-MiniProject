import torch
from torch import nn
from torch.nn.functional import fold, unfold
from .activations import Sigmoid, ReLU
from .sequential import Sequential

torch.set_default_dtype(torch.float64)

class Conv2d():
    def __init__(self, out_channel, in_channel, kernel_size,stride = 1):
        # Batch size?
        self.kernel_size = kernel_size
        self.out_channel = out_channel
        self.stride = stride
        self.input_shape = None
        self.in_channel = in_channel

    
    def set_initial(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def initialize(self, input):
        input_height, input_width = input.size()[2:]
        self.input_shape = (input_height, input_width)

        out_size = lambda s_in, ks, st: int((( s_in - (ks - 1) - 1 ) / st + 1) // 1)
        self.output_shape = (self.out_channel, out_size(input_height, self.kernel_size, self.stride), out_size(input_width, self.kernel_size, self.stride))
        self.weight_shape = (self.out_channel, self.in_channel, self.kernel_size, self.kernel_size)
        self.weight = torch.randn(self.weight_shape)
        self.bias = torch.randn(self.out_channel)
    
    def forward(self, input):
        if self.input_shape == None:
            self.initialize(input)

        #print(self.weight_shape)
        self.input = input
        # change to empty
        #print(self.weight)
        
        # do we need to add padding? 
        unfolded = unfold(self.input, kernel_size= (self.kernel_size,self.kernel_size), stride = self.stride )
        self.output = self.weight.view(self.out_channel, -1) @ unfolded + self.bias.view(1,-1,1)
        self.output = self.output.view(1,self.out_channel,self.input.shape[2] - self.kernel_size + 1, self.input.shape[3] - self.kernel_size + 1)     

        return self.output
    
    def backward(self, output_gradient):
        x, y = output_gradient.size()[-2:]
        ks = self.weight_shape[-1] - 1
        self.output_gradient = output_gradient
        # dL/dK = X_j * output_grad
        inp_dldK = unfold(self.input, kernel_size=self.output_gradient.size()[-2:], stride = self.stride)
        print(self.output_gradient())
        out_dldK = inp_dldK.transpose(1, 2).matmul(self.output_gradient.view(self.output_gradient.size(1) * self.output_gradient.size(2), -1).t()).transpose(1, 2)
        
        
        # dL/db
        # dL/dX_j = sum_i dE/dYi * Kij
        
        
        self.kernel_flipped = self.weight.flip([2,3])
        
        # unstride the output gradient
        zeros = torch.empty((x -1)* (self.stride -1)+x, y + (y-1)* (self.stride -1)).zero_()
        zeros[::self.stride,::self.stride] = output_gradient
        
        self.unstrided_gradient = zeros

        inp_unf = unfold(self.unstrided_gradient, kernel_size=self.kernel_flipped[-2:], stride = 2, padding = (self.kernel_size - 1, self.kernel_size - 1))
        out_unf = inp_unf.transpose(1, 2).matmul(self.kernel_flipped.view(self.kernel_flipped.size(1) * self.kernel_flipped.size(2), -1).t()).transpose(1, 2)
        input_gradient = fold(out_unf, output_size=self.weight_shape[1:], kernel_size=(1, 1), stride=2)

        # Maybe update the weights here??
        self.weight-= self.learning_rate * out_dldK    
        self.bias -= output_gradient

        return input_gradient