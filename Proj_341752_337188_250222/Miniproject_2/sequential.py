from src.module import Module



################### CLASSES ###################



class Sequential(Module):
	"""
		args:
			*args: used only for instantiated layers/activations that we give as input
			
		example:
			model = Sequential(Linear(self.outunits, self.units),
								ReLU(),
								Linear(self.units, self.units),
								ReLU(),
								Linear(self.units, self.units),
								ReLU(),
								Linear(self.units, self.outunits),
								)
			model.forward(x)
			model.backward(x)
			model.reset_gradients()
	"""
	def __init__(self, *args):
		super(Sequential, self).__init__()

		self.vals = []
		self.module = []

		for transf in args:
			self.module.append(transf)

	def __call__(self,x, eval_mode=0):
		return self.forward(x, eval_mode=eval_mode)


	def forward(self,x, eval_mode):
		if not eval_mode: 
			self.vals.append(x)
		for transf in self.module:
			x = transf.forward(x)
			if not eval_mode: 
				self.vals.append(x)
		return x


	def backward(self,x):
		for i in range(len(self.module)-1,-1,-1):
			transf = self.module[i]
			x = transf.backward((x, self.vals[i]))

		# emptying the values after using them
		self.vals = []
		return x

	def reset_gradients(self):
		for mod in self.module:
			if len(mod.param()):
				mod.reset_gradients()
