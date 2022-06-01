from .module import Module

class MSE(Module):
	def __init__(self):
		super(MSE, self).__init__()

	def forward(self, input, target):
		difference = (input - target)
		self.grad = 2.0 * (difference)
		output = difference.pow(2.0)
		self.grad = self.grad / difference.numel()
		output = output.mean()
		return output

	def backward(self):
		gradwrtinput = self.grad
		return gradwrtinput
	
