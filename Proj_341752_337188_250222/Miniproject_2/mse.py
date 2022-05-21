import torch
from activations import *
from module import Module

class MSE(Module):

	def __init__(self):
		super(MSE, self).__init__()
		self.loss = None

	def forward(self,y,y_output):

		N = y.size(0)
		l = 1/N*(y-y_output)**2
		self.loss = l.sum()
		return self.loss

	def backward(self,y,y_output):
		# simply gradient of MSE as a function of y_output
		N = y.size(0)
		return -2/N*(y-y_output)
	