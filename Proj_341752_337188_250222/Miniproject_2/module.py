
# PARENT CLASS
################### CLASSES ###################


class Module:

	def forward(self, *input):
		raise NotImplementedError

	def backward(self, *gradwrtoutput):
		raise NotImplementedError

	def param(self):
		return []


