from .module import Module
import torch
from torch.nn.functional import fold, unfold

torch.set_default_dtype(torch.float64)

class Conv2d(Module):
    def __init__(self, out_channel, in_channel, kernel_size,stride = 1, padding = 0):
        # Batch size?
        super(Conv2d).__init__()
        self.kernel_size = kernel_size
        self.out_channel = out_channel
        self.stride = stride
        self.input_shape = None
        self.in_channel = in_channel
        self.weight_shape = (self.out_channel, self.in_channel, self.kernel_size, self.kernel_size)
        self.padding = padding
        self.weight = torch.empty(self.weight_shape).uniform_(-1/10,1/10)

        self.bias = torch.empty(self.out_channel).zero_()

        self.weight_grad = torch.empty(self.weight_shape).zero_()
        self.bias_grad = torch.empty(self.out_channel).zero_()
        
    def initialize(self, input):
        input_height, input_width = input.size()[2:]
        self.input_shape = (input_height, input_width)
        self.batch_size = input.size(0)

        self.output_shape = (self.out_channel, self.out_size(input_height, self.padding, self.kernel_size, self.stride), self.out_size(input_width,self.padding, self.kernel_size, self.stride))

    def out_size(self, s_in,pad, ks, st):
        return int((( s_in +2*pad - (ks - 1) - 1 ) / st + 1) // 1)
    
    def forward(self, input, eval=False):
        if self.input_shape == None:
            self.initialize(input)
        # if eval == False:
        self.input = input

        unfolded = unfold(input, kernel_size= (self.kernel_size,self.kernel_size), stride = self.stride, padding = self.padding) 
        if eval == False:
            self.unfolded_forward = unfolded
        output = (self.weight.view(self.out_channel, -1)  @ unfolded) 
        output = output.view(input.size(0),self.out_channel,self.output_shape[1], self.output_shape[2] )+ self.bias.view(self.out_channel,1,1)  
        return output # (B, CO, H',W')
    
    def backward(self, gradwrtoutput): # (B, co, )

        batch_size, input_channel, height, width = self.input.size()
        batch_size, output_channel, output_height, output_width = gradwrtoutput.size()
        
        rhs = self.weight.view(output_channel, -1)
        lhs = gradwrtoutput.reshape(batch_size, output_channel, output_height * output_width).transpose(1, 2).reshape(-1, output_channel)
        input = lhs @ rhs
        input = input.view(batch_size, output_height*output_width, -1).transpose(1, 2)
        input_grad = fold(input, kernel_size = self.kernel_size, output_size = (height, width), stride = self.stride, padding = self.padding)

        lhs = gradwrtoutput.view(batch_size, output_channel, -1)
        rhs = self.unfolded_forward.transpose(1, 2)
        self.weight_grad = lhs.bmm(rhs).sum(dim=0)
        self.weight_grad = self.weight_grad.reshape(self.weight_shape)

        self.bias_grad = gradwrtoutput.sum((0,2,3))

        return input_grad.reshape(self.input.shape)

    def param(self):
        return [self.weight, self.bias,self.weight_grad,self.bias_grad]
    
    def zero_grad(self):
        self.weight_grad = self.weight_grad.zero_()
        self.bias_grad = self.bias_grad.zero_()
        
    def step(self, eta):
        self.weight -= eta*self.weight_grad # / self.input.size(0)
        self.bias -= eta*self.bias_grad# / self.input.size(0)