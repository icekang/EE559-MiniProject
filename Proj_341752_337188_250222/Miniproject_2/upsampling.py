import torch
from torch.nn.functional import unfold
from .module import Module
from .convolution import Conv2d


class Upsampling(Module):
    def __init__(self, out_channel, in_channel, kernel_size, stride=1, scale_factor=2):
        super(Upsampling).__init__()
        self.nn = NNUpsample(scale_factor)
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = Conv2d(out_channel, in_channel, kernel_size, self.stride, padding = 1)

    def forward(self, x):
        x = self.nn.forward(x)
        y = self.conv.forward(x)
        return y

    def backward(self, gradwrtoutput):
        x = self.conv.backward(gradwrtoutput)
        return self.nn.backward(x)
    def param(self):
        return [self.conv.weight, self.conv.bias,self.conv.weight_grad,self.conv.bias_grad]
    def zero_grad(self):
        self.conv.weight_grad = self.conv.weight_grad.zero_()
        self.conv.bias_grad = self.conv.bias_grad.zero_()
    def step(self, eta):
        self.conv.step(eta)


    
class NNUpsample(Module):
    def __init__(self, scale_factor):
        super(NNUpsample).__init__()
        self.scale_factor = scale_factor
        self.w = None

    def initialize(self, r):
        self.w = torch.zeros((r.shape[1], r.shape[1], self.scale_factor, self.scale_factor))
        
        for i in range(r.shape[1]):
            self.w[i,i,:,:] = 1

    def forward(self, x):
        self.input_size = x.shape
        self.batch_size = x.size(0)
        self.channels = x.size(1)
        r = x.repeat_interleave(self.scale_factor, dim=2).transpose(2, 3).repeat_interleave(self.scale_factor, dim=2).transpose(2, 3)
        return r

    def backward(self, r):
        if self.w is None:
            self.initialize(r)

        unfolded = unfold(r, kernel_size=(self.scale_factor, self.scale_factor), stride=self.scale_factor)

        lhs = self.w.view(r.shape[1],-1)
        out_unf = lhs @ unfolded
        
        out_unf = out_unf.view(r.shape[0], r.shape[1], self.input_size[2], self.input_size[3])

        return out_unf