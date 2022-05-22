from .module import Module


class Sequential(Module):
	def __init__(self, *args):
		super(Sequential, self).__init__()

		self.modules = []

		for transf in args:
			self.modules.append(transf)

	def __call__(self, x):
		return self.forward(x)


	def forward(self, x):
		for transf in self.modules:
			x = transf.forward(x)
		return x


	def backward(self,x):
		for layer in self.modules[::-1]:
			x = layer.backward(x)
		return x


	def zero_grad(self):
		for layer in self.modules:
			if len(layer.param()):
				layer.zero_grad()
