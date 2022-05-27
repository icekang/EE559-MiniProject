import torch
from .module import Module


class Sigmoid(Module):
    def __init__(self):
        super(Sigmoid).__init__()
    
    def build(self,x):
        self.input_shape = x.size()
        self.output_shape = x.size()
        self.built = True
    
    def __call__(self,x):
        return self.forward(x)

    def forward(self,x):
        self.output = 1.0 / ( 1 + (-x).exp() )
        #print("self.output",self.output)
        return self.output

    def _grad(self, x):

        return self.forward(x) * (1-self.forward(x))

    def backward(self, gradwrtoutput):
        #print("self._grad(gradwrtoutput)",self._grad(gradwrtoutput))
        return  self._grad(gradwrtoutput)
    


class ReLU(Module):

    def __init__(self):
        super(ReLU).__init__()

    def __call__(self,x):
        return self.forward(x)

    def forward(self, x):
        self.input = (x > 0).float() * x
        return self.input

    def _grad(self, x):
        return (x > 0).float()

    def backward(self, gradwrtoutput):
        return gradwrtoutput * self._grad(self.input)