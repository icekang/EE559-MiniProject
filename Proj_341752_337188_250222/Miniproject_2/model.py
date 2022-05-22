import torch
from .activations import Sigmoid, ReLU
from .sequential import Sequential
from .module import Module, Conv2d, Upsampling

torch.set_default_dtype(torch.float64)

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
        
        # Conv2d need to be initialized by passing dummy input
        # self.forward(torch.zeros((1, 3, 512, 512)))
    
    def forward(self, x):
        # x = x.float()
        return self.model(x)
    
    def predict(self, test_input):
        test_input = test_input / 255.0
        output = self.forward(test_input)
        output = output * 255.0
        return torch.clip(output, 0.0, 255.0)

    def backward(self, x):
        return self.model.backward(x)
    
    def load_pretrained_model(self):
        self.model.load_pretrained_model()