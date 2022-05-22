import torch
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
    
    def forward(self, x):
        # x = x.float() /255.0
        return self.model(x)
    
    def predict(self, test_input):
        # test_input = test_input.float() / 255.0
        output = self.forward(test_input)
        output = output * 255.0
        return torch.clip(output, 0.0, 255.0)

    def backward(self, x):
        return self.model.backward(x)


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

        # print(input_height)
        # print(input_width)
        # print(self.kernel_size)
        self.output_shape = (self.out_channel, self.out_size(input_height, self.kernel_size, self.stride), self.out_size(input_width, self.kernel_size, self.stride))
        self.weight_shape = (self.out_channel, self.in_channel, self.kernel_size, self.kernel_size)
        self.weight = torch.ones(self.weight_shape)
        self.bias = torch.zeros(self.out_channel)
    
    def out_size(self, s_in, ks, st):
        return int((( s_in - (ks - 1) - 1 ) / st + 1) // 1)
    
    def forward(self, input):
        if self.input_shape == None:
            self.initialize(input)
        # input = input.float()

        self.input = input
        
        unfolded = unfold(self.input, kernel_size= (self.kernel_size,self.kernel_size), stride = self.stride )
        print('unfolded.dtype', unfolded.dtype)
        print('self.bias.dtype', self.bias.dtype)
        print('self.weight.view(self.out_channel, -1).dtype', self.weight.view(self.out_channel, -1).dtype)
        self.output = self.weight.view(self.out_channel, -1) @ unfolded + self.bias.view(1,-1,1)
        self.output = self.output.view(input.size(0),self.out_channel,self.output_shape[1], self.output_shape[2])     
        
        return self.output
    
    def backward(self, output_gradient):
        x, y = output_gradient.size()[-2:]
        ks = self.weight_shape[-1] - 1
        self.output_gradient = output_gradient
        # dL/dK = X_j * output_grad
        #inp_dldK = unfold(self.input, kernel_size=self.output_gradient.size()[-2:], stride = self.stride)
        #out_dldK = inp_dldK.transpose(1, 2).matmul(self.output_gradient.view(self.output_gradient.size(1) * self.output_gradient.size(2), -1).t()).transpose(1, 2)
        
        
        # dL/db
        # print("self.weights.size()",self.weight.size())
        # dL/dX_j = sum_i dE/dYi * Kij
        
        # self.kernel_flipped = self.weight.permute([1,0,2,3])
        self.kernel_flipped = self.weight.flip([2,3])
        
        # unstride the output gradient
        
        # second index of zeros is output channels
        
        zeros = torch.empty(self.input.size(0),self.out_channel,(x-1)* (self.stride-1)+x, y + (y-1)* (self.stride -1)).zero_()
        zeros[:,:,::self.stride,::self.stride] = output_gradient
        
        self.unstrided_gradient = zeros
        print('self.unstrided_gradient.size()', self.unstrided_gradient.size())
        
        
        unfolded = unfold(self.unstrided_gradient, kernel_size= (self.kernel_size,self.kernel_size), stride = 1, padding = (self.kernel_size - 1, self.kernel_size - 1))
        print(unfolded)
        print('unfolded.size()', unfolded.size())
        
        lhs = self.kernel_flipped.view(self.in_channel, self.kernel_size ** 2 * self.out_channel)
        print('lhs.size()', lhs.size())
        self.input_grad = lhs @ unfolded
        
        #self.input_grad = fold(self.input_grad)
        
        self.input_grad = self.input_grad.view(self.input.size(0),self.in_channel,self.input_shape[0], self.input_shape[1])     
        print(self.input_grad.size())


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
        self.nn = NNUpsample(scale_factor)
        self.padding = Padding((kernel_size - 1) // 2)
        self.conv = Conv2d(out_channel, in_channel, kernel_size, stride)

    def forward(self, x):
        x = self.nn.forward(x)
        x = self.padding.forward(x)
        return self.conv.forward(x)

    def backward(self, gradwrtoutput):
        x = self.nn.backward(gradwrtoutput)
        x = self.padding.backward(x)
        return self.conv.backward(x)
    

class Padding(Module):
    def __init__(self, padding):
        super(Padding).__init__()
        self.padding = padding

    def forward(self, x):
        padded = torch.zeros((x.size(0), x.size(1), x.size(2) + 2 * self.padding, x.size(3) + 2 * self.padding))
        padded[:, :, self.padding:x.size(2) + self.padding, self.padding:x.size(3) + self.padding] = x
        return padded

    def backward(self, x):
        return x[:, :, self.padding:-self.padding, self.padding:-self.padding]