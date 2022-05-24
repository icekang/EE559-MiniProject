from black import Mode
import torch
from torch import nn
from torch.nn.functional import fold, unfold
from .activations import Sigmoid, ReLU
from .sequential import Sequential
from .module import Module
import floor

torch.set_default_dtype(torch.float64)

class Model():
    def __init__(self):
        self.model = Sequential(
            Conv2d(48, 3, 2, 2),
            ReLU(),
            Conv2d(96, 48, 2, 2),
            ReLU(),
            Upsampling(48, 96, 3, 1, 2),
            ReLU(),
            Upsampling(3, 48, 3, 1, 2),
            Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)
    
    def predict(self, test_input):
        test_input = test_input.float() / 255.0
        output = self.forward(test_input)
        output = output * 255.0
        return torch.clip(output, 0.0, 255.0)

    def backward(self, x):
        return self.model.backward(x)

import torch
from torch import nn
from torch.nn.functional import fold, unfold

import torch
from torch import nn
from torch.nn.functional import fold, unfold

import torch
from torch import nn
from torch.nn.functional import fold, unfold
import math


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

        print(input_height)
        print(input_width)
        print(self.kernel_size)
        self.output_shape = (self.out_channel, self.out_size(input_height, self.kernel_size, self.stride), self.out_size(input_width, self.kernel_size, self.stride))
        self.weight_shape = (self.out_channel, self.in_channel, self.kernel_size, self.kernel_size)
        self.weight = torch.empty(self.weight_shape).normal_()
        self.bias = torch.empty(self.out_channel).normal_()
    
    def padding(self,input_size, ks, stride):
        quotient = math.ceil(input_size / stride)
        a =   (((input_size - stride+1) % ks)) % stride
        return a

    
    def out_size(self, s_in, ks, st):
        return int((( s_in - (ks - 1) - 1 ) / st + 1) // 1)
    
    def forward(self, input):
        if self.input_shape == None:
            self.initialize(input)
        input = input.float()

        self.input = input
        
        unfolded = unfold(self.input, kernel_size= (self.kernel_size,self.kernel_size), stride = self.stride)
        self.output = self.weight.view(self.out_channel, -1) @ unfolded + self.bias.view(1,-1,1)
        self.output = self.output.view(input.size(0),self.out_channel,self.output_shape[1], self.output_shape[2])     
        
        return self.output
    
    def backward(self, output_gradient):
        
        learning_rate = 0.01
        
        ## define some vars that I might delete
        x, y = output_gradient.size()[-2:]
        ks = self.weight_shape[-1] - 1
        self.output_gradient = output_gradient

        
        self.weight_grads = torch.empty(self.weight_shape).zero_()
        self.bias_grads = torch.empty(self.out_channel).zero_()
        
        
        ### CALCULATE dL/dK
        zeros = torch.empty(self.input.size(0),self.input.size(1), self.input.size(2) - self.padding(self.input.size(2), self.output_shape[1], self.stride), self.input.size(3) - self.padding(self.input.size(3), self.output_shape[2], self.stride)).zero_()
        zeros[:,:,:self.input.size(2),:self.input.size(3)] = self.input[:,:,:self.input.size(2) - self.padding(self.input.size(2), self.output_shape[1], self.stride), : self.input.size(3) -self.padding(self.input.size(3), self.output_shape[2], self.stride)]
        
 
        self.input2 = zeros
        
        
        unfolded = unfold(self.input2.view(self.in_channel,1,  self.input2.size(2), self.input2.size(3)), kernel_size = self.output_shape[1:], dilation = self.stride, padding = 0, stride = 1)

        wxb = output_gradient.view(self.out_channel,-1) @ unfolded

        actual = wxb.view(self.out_channel, self.in_channel,self.kernel_size, self.kernel_size) 
        self.weight_grads += actual
                
        size_grad = self.output_gradient.size()[-2:]       
        
        ### CALCULATE dL/dX
        
        # we will flip the kernel to do the full convolution
        self.kernel_flipped = self.weight.flip([2,3])
        
        # unstride the output gradient        
        zeros = torch.empty(self.input.size(0),self.out_channel,(x-1)* (self.stride-1)+x + self.padding(self.input.size(2), self.output_shape[1], self.stride) , y + (y-1)* (self.stride -1) + self.padding(self.input.size(2), self.output_shape[1], self.stride)).zero_()
        zeros[:,:,::self.stride,::self.stride] = output_gradient
        
        self.unstrided_gradient = zeros        
        
        
        unfolded = unfold(self.unstrided_gradient, kernel_size= self.kernel_size, stride = 1, padding = (self.kernel_size - 1, self.kernel_size - 1))
        
        lhs = self.kernel_flipped.view(self.in_channel, self.kernel_size ** 2 * self.out_channel)
        self.input_grad = lhs @ unfolded
                
        self.input_grad = self.input_grad.view(self.input.size(0),self.in_channel,self.input_shape[0], self.input_shape[1])     

        # CALCULATE dL/dB
        
        self.bias_grads += self.output_gradient.mean((0,2,3))

        self.weight -= learning_rate * self.weight_grads
        self.bias -= learning_rate * self.bias_grads

        return self.input_grad

class NNUpsample(Module):
    def __init__(self, scale_factor):
        super(NNUpsample).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        r = x.repeat_interleave(self.scale_factor, dim=2).transpose(2, 3).repeat_interleave(2, dim=2).transpose(2, 3)
        return r

    def backward(self, r):
        scale = self.scale_factor
        res = []
        for c in range(r.shape[1]): # aggregrate by channel
            w = torch.zeros((1, 3, scale, scale))
            w[:, c, :, :] = 1
            unfolded = unfold(r.float(), kernel_size=(scale, scale), stride=scale)
            out_unf = unfolded.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
            res.append(out_unf[:, 0, :].reshape((3, 3)))
        res = torch.dstack(res)
        return res


class Upsampling(Module):
    def __init__(self, out_channel, in_channel, kernel_size, stride=1, scale_factor=2):
        super(Upsampling).__init__()
        self.conv = Conv2d(out_channel, in_channel, kernel_size, stride)
        self.nn = NNUpsample(scale_factor)

    def forward(self, x):
        return self.conv(self.nn(x))

    def backward(self, gradwrtoutput):
        x = self.nn.backward(gradwrtoutput)
        return self.conv.backward(x)