import torch

from loss import *
from activations import *

from utils import *
from model import *
from generate import *



################### FUNC ###################



def train(model, train_input, train_label, criterion, nb_epochs,
			mini_batch_size, eta, loss_flag=0, verbose=1):
	"""
		args:
			model: MLP we want to train
			train_input: inputs of coordinates Nx2
			train_label: inputs for ground truth, points belong to circle or not (1 for belonging)
			criterion: MSE
			nb_epochs: number of epochs
			mini_batch_size: number of datapoints used for one weight update
			eta: learning rate for weight update
			loss_flag: set to 0 when using MSE and set to 1 when using CROSSENTROPY
			verbose: flag to print logs during training 
			
		example:
			model = Net(hidden=25, activation='tanh')
			losses, accs = train(model, ...)
	"""

	losses = []
	accs = []
	for e in range(nb_epochs):

		full_loss = 0

		for b in range(0, train_input.size(1), mini_batch_size):
			train_batch = train_input[:,b:b+mini_batch_size]

			output = model.forward(train_batch) # (2,batch_size)
			ground_truth = train_label[:,b:b+mini_batch_size] # (2,batch_size)
			if loss_flag:
				ground_truth = torch.argmax(ground_truth,axis=1)

			# reset gradients before propagating
			#model.reset_gradients()

			# compute loss & # backprop the loss
			loss = criterion.forward(ground_truth, output) 
			full_loss += loss
			print("full_loss", full_loss)
			model.backward(criterion.backward(ground_truth, output))


			# optim 
			#for i in range(len(model.full.module)):
			#	layer = model.full.module[i]
			#	if not len(layer.param()): 
					# case of activation, no need to update any params
			#		continue

				#layer.weight -= eta * layer.dldw
				#layer.bias -= eta * layer.dldb 


		if  (e % 10 == 0): 
			# eval
			output = model.forward(train_input)
			if verbose:
				print('Epoch {}: train loss-> {}'.format(e,full_loss/train_input.size(1)))

			losses.append(full_loss/train_input.size(1))
			#accs.append(acc)

	return losses, accs

