import torch
from .module import Module
from .activations import Sigmoid, ReLU
from .sequential import Sequential
from .convolution import Conv2d
from .upsampling import Upsampling
from .mse import MSE


class Model(Module):
    def __init__(self):
        self.model = Sequential(
            Conv2d(48, 3, 2, 2),
            ReLU(),
            Conv2d(96, 48, 2, 2),
            ReLU(),
            Upsampling(48, 96, 3, 1, 2),
            ReLU(),
            Upsampling(3, 48, 3, 1, 2),
            Sigmoid()
        )
        self.criterion = MSE()
    

    def load_pretrained_model(self):
        self.model.load_pretrained_model()

    def train( self, train_input, train_target, num_epochs ) -> None :
        losses = []
        mini_batch_size = 100
        lr = 0.0001

        train_input = train_input.double() / 255.0
        train_target = train_target.double() / 255.0
        for e in range(num_epochs):
            full_loss = 0
            for b in range(0, train_input.size(0), mini_batch_size):
                #optimizer.zero_grad()
                train_batch = train_input[b:b+mini_batch_size,:,:,:]
                self.model.zero_grad()
                output = self.forward(train_batch, eval=False) # (2,batch_size)
                ground_truth = train_target[b:b+mini_batch_size,:,:,:] # (2,batch_size)

                loss = self.criterion.forward(output, ground_truth)
                full_loss += loss * mini_batch_size
                loss_grad = self.criterion.backward()
                self.backward(loss_grad)
                for i in range(len(self.model.modules)):
                    layer = self.model.modules[i]
                    if not len(layer.param()): 
                        # case of activation, no need to update any params
                        continue
                    layer.step(lr)

            if  (e % 5 == 0): 
                # eval
                # output = self.forward(train_input)
                print('Epoch {}: train loss-> {}'.format(e,full_loss/train_input.size(0)))

            losses.append(full_loss / train_input.size(0))

        return losses

    def forward(self, x, eval=False):
        return self.model(x, eval)
    
    def predict(self, test_input):
        test_input = test_input / 255.0
        output = self.forward(test_input, eval=True)
        output = output * 255.0
        return torch.clip(output, 0.0, 255.0)

    def backward(self, x):
        return self.model.backward(x)