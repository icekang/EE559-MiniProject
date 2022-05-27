from .module import Module
import torch
from torch.nn.functional import fold, unfold

torch.set_default_dtype(torch.float64)

class Conv2d(Module):
    
    def __init__(self, out_channel, in_channel, kernel_size,stride = 1, padding = 0):
        super(Conv2d).__init__()
        self.kernel_size = kernel_size
        self.out_channel = out_channel
        self.stride = stride
        self.input_shape = None
        self.in_channel = in_channel
        self.weight_shape = (self.out_channel, self.in_channel, self.kernel_size, self.kernel_size)
        self.padding_ = padding
        self.weight = torch.empty(self.weight_shape).uniform_(-1/10,1/10)

        self.bias = torch.empty(self.out_channel).normal_()

        self.weight_grad = torch.empty(self.weight_shape).zero_()
        self.bias_grad = torch.empty(self.out_channel).zero_()
        
    def initialize(self, input):
        self.input = input
        input_height, input_width = self.input.size()[2:]
        self.input_shape = (input_height, input_width)
        self.batch_size = self.input.size(0)

        self.output_shape = (self.out_channel, self.out_size(input_height, self.padding_, self.kernel_size, self.stride), self.out_size(input_width,self.padding_, self.kernel_size, self.stride))

    def out_size(self, s_in,pad, ks, st):
        return int((( s_in +2*pad - (ks - 1) - 1 ) / st + 1) // 1)
    
    def forward(self, input):
        if self.input_shape == None:
            self.initialize(input)

        unfolded = unfold(self.input, kernel_size= (self.kernel_size,self.kernel_size), stride = self.stride, padding = self.padding_)
        
        self.unfolded_forward = unfolded
        
        self.output = (self.weight.view(self.out_channel, -1)  @ unfolded)  + self.bias.view(1,-1,1)

        self.output = self.output.view(self.input.size(0),self.out_channel,self.output_shape[1], self.output_shape[2] )
        
        return self.output # (B, CO, H',W')
    
    def backward(self, output_gradient):
#         output_gradient = output_gradient / self.input.size(0)
        #print("OUTPUT_GRADIENT.MAX()", output_gradient.min())
#         output_gradient[(output_gradient  >100)] = 100
#         output_gradient[(output_gradient < -100)] = -100
        
        ### CALCULATE dL/dK

        B, ci, h, w = self.input.size()
        B, co, oh, ow = output_gradient.size()
        ks = self.kernel_size

        # print('self.weight.dtype, output_gradient.dtype', self.weight.dtype, output_gradient.dtype)
        input =  self.weight.view(-1, co) @ output_gradient.view(B,co,-1)
        input_grad = fold(input, kernel_size = ks, output_size = (h,w),stride = self.stride,padding = self.padding_)

#         self.weight_grad = output_gradient.view(B, co, -1).bmm(self.unfolded_forward.transpose(1,2)).sum(dim = 0).reshape(self.weight_shape) 
        self.weight_grad = output_gradient.view(B, co, -1).bmm(self.unfolded_forward.transpose(1,2)).sum(dim = 0).reshape(self.weight_shape) / output_gradient.numel() # (co, ci, ks, ks)  (B,co, output_size, output_size) (B, ci, in_size, in_size)

        #print("self.weight_grad.mean()",self.weight_grad.mean())
        self.bias_grad = output_gradient.mean((0,2,3))

#         self.bias_grad = output_gradient.sum((0,2,3))
        #print("self.bias_grad.mean()",self.bias_grad.mean())
        return input_grad

    def param(self):
        return [self.weight, self.bias,self.weight_grad,self.bias_grad]
    
    def zero_grad(self):
        self.weight_grad = self.weight_grad.zero_()
        self.bias_grad = self.bias_grad.zero_()
        
    def step(self, eta):
        self.weight -= eta*self.weight_grad # / self.input.size(0)
        self.bias -= eta*self.bias_grad# / self.input.size(0)