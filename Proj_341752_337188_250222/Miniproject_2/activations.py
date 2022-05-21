import torch
from module import Module



################### CLASSES ###################



class ReLU(Module):
	
	def __init__(self):
		super(ReLU).__init__()

	def __call__(self,x):
		return self.forward(x)

	def forward(self,x):
		return (x > 0).float() * x

	def _grad(self,x):
		return (x > 0).float()

	def backward(self,x):
		# compute dldsl 
		dldxl, s = x
		return dldxl * self._grad(s) # hadamard prod
		

