import torch
from torch.nn.functional import fold, unfold
import numpy as np

def Conv():
    def __init__(self, input_shape, kernel_size, depth, stride):
        # Batch size?
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.stride = stride
        out_size = lambda s_in, ks, st: (( s_in - (ks - 1) - 1 ) / st + 1) // 1
        self.output_shape = (depth, out_size(input_height, kernel_size, self.stride), out_size(input_width, kernel_size, self.stride))
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        inp_unf = unfold(self.input, kernel_size=self.kernels_shape[-2:], stride=2)
        out_unf = inp_unf.transpose(1, 2).matmul(self.kernels.view(self.kernels.size(0), -1).t()).transpose(1, 2)
        self.output = fold(out_unf, output_size=self.output_shape[1:], kernel_size=(1, 1), stride=2)
        self.output += self.biases

        return self.output

    def backward(self, output_gradient):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        inp_unf = unfold(self.input, kernel_size=self.kernels_shape[-2:], stride=2)
        out_unf = inp_unf.transpose(1, 2).matmul(output_gradient.view(output_gradient.size(0), -1).t()).transpose(1, 2)
        kernels_gradient = fold(out_unf, output_size=self.kernels_shape[1:], kernel_size=(1, 1), stride=2)

        ks = self.kernels_shape[-1] - 1
        inp_unf = unfold(np.pad(output_gradient, ((0, 0), (0, 0), (ks, ks), (ks, ks))), kernel_size=self.kernels_shape[-2:], stride=2)
        out_unf = inp_unf.transpose(1, 2).matmul(self.kernels.view(self.kernels.size(0), -1).t()).transpose(1, 2)
        input_gradient = fold(out_unf, output_size=self.kernels_shape[1:], kernel_size=(1, 1), stride=2)

        # Maybe update the weights here??

        return input_gradient