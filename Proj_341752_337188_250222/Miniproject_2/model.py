import torch
from torch import nn
from torch.nn.functional import fold, unfold
from .activations import Sigmoid, ReLU
from .sequential import Sequential
from .module import Module

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
    

    def load_pretrained_model(self):
        self.model.load_pretrained_model()


    def forward(self, x):
        return self.model(x)
    
    def predict(self, test_input):
        test_input = test_input / 255.0
        output = self.forward(test_input)
        output = output * 255.0
        return torch.clip(output, 0.0, 255.0)

    def backward(self, x):
        return self.model.backward(x)


import math


class Conv2d():
    
    def __init__(self, out_channel, in_channel, kernel_size,stride = 1, padding = 0):
        # Batch size?
        self.kernel_size = kernel_size
        self.out_channel = out_channel
        self.stride = stride
        self.input_shape = None
        self.in_channel = in_channel
        self.weight_shape = (self.out_channel, self.in_channel, self.kernel_size, self.kernel_size)
        self.padding_ = padding
        self.weight = torch.empty(self.weight_shape,device=device).normal_()
        self.bias = torch.empty(self.out_channel,device =device).normal_()
        
    def initialize(self, input):
        input_height, input_width = input.size()[2:]
        self.input_shape = (input_height, input_width)
        self.batch_size = input.size(0)
        self.output_shape = (self.out_channel, self.out_size(input_height, self.kernel_size, self.stride), self.out_size(input_width, self.kernel_size, self.stride))


    
    def padding(self, input_size, ks, stride):
        j = None
        for i in range(10):
            k = (input_size - ks + i) / stride
            if k.is_integer() == True:
                j = i
                break
        return j
    
    def out_size(self, s_in, ks, st):
        return int((( s_in - (ks - 1) - 1 ) / st + 1) // 1)
    
    def forward(self, input):
        if self.input_shape == None:
            self.initialize(input)
        input = input.float()
        self.input = input
        st_pad = self.padding(self.input.size(2), self.kernel_size, self.stride) 

        #self.padding_ = padding(self.input_shape[0], self.kernel_size, self.stride)
        zeros = torch.empty(self.input.size(0),self.input.size(1), self.input.size(2) + st_pad, self.input.size(3) + st_pad, device = device).zero_().float()
        zeros[:,:,:self.input.size(2),:self.input.size(3)] = self.input[:,:,:self.input.size(2), : self.input.size(3)]
        input = zeros
        
        print("self.output_shape",self.output_shape)
        #st_pad = 1
        print("st_pad",st_pad)
        unfolded = unfold(self.input, kernel_size= (self.kernel_size,self.kernel_size), stride = self.stride, padding = 0).float() 
        self.output = self.weight.view(self.out_channel, -1).float()  @ unfolded.float()  + self.bias.view(1,-1,1).float() 

        self.output = self.output.view(input.size(0),self.out_channel,self.output_shape[1], self.output_shape[2] ).float() 
        
        return self.output
    
    def backward(self, output_gradient):
        print("output_gradient.size()",output_gradient.size())
        learning_rate = 0.01
        
        ## define some vars that I might delete
        x, y = output_gradient.size()[-2:]
        ks = self.weight_shape[-1] - 1
        self.output_gradient = output_gradient
        print("output_gradient.size()",output_gradient.size())
        
        self.weight_grads = torch.empty(self.weight_shape,device=device).zero_()
        self.bias_grads = torch.empty(self.out_channel,device=device).zero_()
        
        
        ### CALCULATE dL/dK
        st_pad = self.padding(self.input.size(2), self.output_shape[1], self.stride) 
        st_pad2 = self.padding(self.input.size(3), self.output_shape[2], self.stride)
        
        zeros = torch.empty(self.input.size(0),self.input.size(1), self.input.size(2) + st_pad, self.input.size(3) + st_pad2, device = device).zero_().float()
        zeros[:,:,:self.input.size(2),:self.input.size(3)] = self.input[:,:,:self.input.size(2), : self.input.size(3)]
        self.input2 = zeros
        unfolded = unfold(self.input2.view(self.in_channel,self.input.size(0),  self.input2.size(2), self.input2.size(3)), kernel_size = self.output_shape[1:], dilation = self.stride, padding = st_pad, stride =1)
        print("unfolded.size()",unfolded.size())
        print("output_gradient.view(self.out_channel,-1).size()",output_gradient.view(self.out_channel, self.batch_size,-1).size())
        
        wxb = output_gradient.view(self.out_channel,-1) @ unfolded#.view(self.in_channel,self.out_channel, -1)
        
        print("wxb.size()",wxb.size())
        #print("wxb.size()", wxb.size())
        
        actual = wxb.view(self.out_channel, self.in_channel,self.kernel_size, self.kernel_size).float()

        self.weight_grads += actual
                
        size_grad = self.output_gradient.size()[-2:]       
        
        # we will flip the kernel to do the full convolution
        self.kernel_flipped = self.weight.flip([2,3])
        
        # unstride the output gradient    
        zeros = torch.empty(self.input.size(0),self.out_channel,(x-1)* (self.stride-1)+x + st_pad , y + (y-1)* (self.stride -1) + st_pad2).zero_()
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
    

import math
class Upsampling(Module):
    def __init__(self, out_channel, in_channel, kernel_size, stride=1, scale_factor=2):
        super(Upsampling).__init__()
        self.nn = NNUpsample(scale_factor)
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = Conv2d(out_channel, in_channel, kernel_size, stride)

    def forward(self, x):
        pad = self.padding2(x.size(2), self.kernel_size, self.stride)
        
        self.padding = Padding(1)
        x = self.nn.forward(x)
        print("upsampling after nn size", x.size())
        print("pad", pad)
        x = self.padding.forward(x)
        print("after padding size", x.size())
        y = self.conv.forward(x)
        print("upsampling after conv size", y.size())
        return y

    def backward(self, gradwrtoutput):
        x = self.conv.backward(gradwrtoutput)
        x = self.padding.backward(x)
        return self.nn.backward(x)
    def padding2(self, input_size, ks, stride):
        j = None
        for i in range(10):
            k = (input_size - ks + i) / stride
            if k.is_integer() == True:
                j = i
                break
        return j
    
class Padding(Module):
    def __init__(self, padding):
        super(Padding).__init__()
        self.padding = padding

    def forward(self, x):
        padded = torch.zeros((x.size(0), x.size(1), x.size(2) + 2 * self.padding, x.size(3) + 2 * self.padding))
        padded[:, :, self.padding:x.size(2) + self.padding, self.padding:x.size(3) + self.padding] = x
        return padded

    def backward(self, x):    
        y = x[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return y
    

    

class NNUpsample(Module):
    def __init__(self, scale_factor):
        super(NNUpsample).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        self.input_size = x.size()
        self.batch_size = x.size(0)
        self.channels = x.size(1)
        r = x.repeat_interleave(self.scale_factor, dim=2).transpose(2, 3).repeat_interleave(self.scale_factor, dim=2).transpose(2, 3)
        return r

    def backward(self, r):
        
        scale = self.scale_factor
        res = []
        for c in range(r.shape[1]): # aggregrate by channel
            w = torch.zeros((self.batch_size, r.shape[1], scale, scale), device = device).float()
            w[:, c, :, :] = 1
            unfolded = unfold(r.float(), kernel_size=(scale, scale), stride=scale).float()

            lhs = w.view(self.batch_size, r.shape[1],-1)
            unfolded = unfolded.view(self.batch_size, scale**2,-1)
            #print("unfolded.shape",unfolded.shape)

            #print("lhs.shape", lhs.shape)
            out_unf = lhs @ unfolded
            
            #out_unf = unfolded.transpose(1, 2).matmul(w.view(r.shape[1], -1).t()).transpose(1, 2)
            #print("out_unf[:, 0, :].reshape((self.batch_size,r.shape[1],self.input_size[2], self.input_size[3]))",out_unf[:, 0, :].reshape((self.batch_size,r.shape[1],self.input_size[2], self.input_size[3])).sum(1, keepdim=True).shape)
            res.append(out_unf[:, c, :].reshape((self.batch_size,r.shape[1],self.input_size[2], self.input_size[3])).sum(1, keepdim = True))
        res = torch.stack(res, dim = 1).sum(2)
        return res

    