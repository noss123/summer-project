import torch
import torch.nn as nn

def drop_path(x, drop_prob=0.0, is_training=False):
    # do not drop if testing
    if drop_prob == 0.0 or not is_training:
        return x
    keep_prob = 1 - drop_prob
    # random tensor of shape [batchsize, 1, 1] to broadcast
    # this will zero out individual image tensors with probability drop_prob
    shape = (x.shape[0],) + (x.ndim-1) * (1,)
    # .new_empty() is a method here so that it can clone datatype and location (device) of x
    keep_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    # inverted dropout scaling to maintain expected value of tensor
    if keep_prob > 0.0:
        keep_tensor.div_(keep_prob)

    return x * keep_tensor

class StochasticDepth(nn.Module):
    def __init__(self, drop_path_prob):
        super().__init__()
        self.drop_path_prob = drop_path_prob

    def forward(self, x):
        # the self.training attr is set to True by default on initialising nn.Module
        # set to False when model.eval() is called, and set to True when model.train() is called
        return drop_path(x, self.drop_path_prob, self.training)