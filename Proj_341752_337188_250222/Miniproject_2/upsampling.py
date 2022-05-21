import torch
from torch import nn
from torch.nn.functional import fold, unfold

class NNUpsample(Module):
    def __init__(self, array, scale_factor):
        super(Upsample, self).__init__()
        self.array = array
        self.scale_factor = scale_factor

    def forward(self):
        r = self.array.repeat_interleave(2, dim=2).transpose(2, 3).repeat_interleave(2, dim=2).transpose(2, 3)
        return r

    def backward(self, r):
        scale = self.scale_factor
        res = []
        for c in range(r.shape[1]): # aggregrate by channel
            w = torch.zeros((1, 3, scale, scale))
            w[:, c, :, :] = 1
            unfolded = unfold(r.float(), kernel_size=(scale, scale), stride=scale)
            # print(unfolded)
            # print(w)
            out_unf = unfolded.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
            # print(out_unf)
            # break
            res.append(out_unf[:, 0, :].reshape((3, 3)))
        res = torch.dstack(res)
        res
