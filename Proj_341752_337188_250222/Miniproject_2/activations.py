import torch
from .module import Module


class Sigmoid(Module):
	def __init__(self):
		super(Sigmoid).__init__()
    
	def build(self,x):
		self.input_shape = x.size()
		self.output_shape = x.size()
		self.built = True
    
	def _call_(self,x):
		return self.forward(x)

	def forward(self, x, eval=False):
		if not eval:
			self.input = x
		output = 1.0 / ( 1 + (-x).exp() )
		#print("self.output",self.output)
		return output

	def _grad(self, x):
		return self.forward(x) * (1-self.forward(x))

	def backward(self, gradwrtoutput):
		#print("self._grad(gradwrtoutput)",self._grad(gradwrtoutput))
		return  gradwrtoutput * self._grad(self.input)
    


class ReLU(Module):

    def __init__(self):
        super(ReLU).__init__()

    def __call__(self, x, eval=False):
        return self.forward(x)

    def forward(self, x, eval=False):
        input = (x > 0) * x
        return input

    def _grad(self, x):
        return (x > 0)

    def backward(self, gradwrtoutput):
        return gradwrtoutput * self._grad(gradwrtoutput)