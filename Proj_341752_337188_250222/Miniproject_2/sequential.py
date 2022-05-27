import torch
from .module import Module
from .convolution import Conv2d
from .upsampling import  Upsampling
from pathlib import Path

class Sequential(Module):
	def __init__(self, *args):
		super(Sequential, self).__init__()

		self.modules = []

		for layer in args:
			self.modules.append(layer)

	def __call__(self, x, eval=False):
		return self.forward(x, eval)


	def forward(self, x, eval=False):
		for layer in self.modules:
			x = layer.forward(x, eval)
		return x


	def state_dict(self):
		state = list()
		for layer in self.modules:
			if isinstance(layer, Conv2d):
				state.append({'weight': layer.weight, 'bias': layer.bias})
			elif isinstance(layer, Upsampling):
				state.append({'weight': layer.conv.weight, 'bias': layer.conv.bias})
			else:
				state.append({})
		return state

	def load_pretrained_model(self):
        ## This loads the parameters saved in bestmodel.pth into the model
		model_path = Path(__file__).parent / "bestmodel.pth"
		state_dict = torch.load(model_path)

		for i, layer in enumerate(self.modules):
			if isinstance(layer, Conv2d):
				layer.weight.data = state_dict[i]['weight']
				layer.bias.data = state_dict[i]['bias']
			elif isinstance(layer, Upsampling):
				layer.conv.weight.data = state_dict[i]['weight']
				layer.conv.bias.data = state_dict[i]['bias']


	def backward(self,x):
		for layer in self.modules[::-1]:
			x = layer.backward(x)
		return x


	def zero_grad(self):
		for layer in self.modules:
			if len(layer.param()):
				layer.zero_grad()
