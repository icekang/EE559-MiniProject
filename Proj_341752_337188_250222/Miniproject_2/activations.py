import torch
from .module import Module



################### CLASSES ###################



class Sigmoid(Module):
	def __init__(self):
		super(Sigmoid).__init__()
	
	def __call__(self,x):
		return self.forward(x)
	
	def forward(self,x):
		return 1 / ( 1 + torch.exp(-x) )
	
	def _grad(self, x):
		return self.forward(x) * (1-self.forward(x))
	
	def backward(self, gradwrtoutput):
		return self._grad(gradwrtoutput)


class ReLU(Module):
	
	def __init__(self):
		super(ReLU).__init__()

	def __call__(self,x):
		return self.forward(x)

	def forward(self, x):
		return (x > 0).float() * x

	def _grad(self, x):
		return (x > 0).float()

	def backward(self, gradwrtoutput):
		return self._grad(gradwrtoutput)