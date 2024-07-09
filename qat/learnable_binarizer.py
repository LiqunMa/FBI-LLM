import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from .base_binarizer import STEBinary, BinaryInterface

class BinaryLinearWscales(nn.Module, BinaryInterface):
    def __init__(self, weight, bias, scaling_pattern, init_method = 'xnor') -> None:
        super().__init__()
        self.weight = nn.Parameter(weight.to(torch.float32).data)
        if bias is not None:
            self.bias = nn.Parameter(bias.to(torch.float32).data)
        else:
            self.bias = None
        self.scaling_pattern = scaling_pattern

        if self.scaling_pattern == 'row':
            self.wscale = nn.Parameter(torch.zeros(self.weight.shape[0], 1), requires_grad=True)
            self.wbias = nn.Parameter(torch.zeros(self.weight.shape[0], 1), requires_grad=True)
            if init_method == 'xnor':
                mean = self.weight.mean(-1, keepdim=True)
                scale = (self.weight - mean).abs().mean(-1, keepdim=True)
                self.wscale.data = scale
                self.wbias.data = mean
        elif self.scaling_pattern == 'column':
            self.wscale = nn.Parameter(torch.zeros(1, self.weight.shape[1]), requires_grad=True)
            self.wbias = nn.Parameter(torch.zeros(1, self.weight.shape[1]), requires_grad=True)
            if init_method == 'xnor':
                mean = self.weight.mean(0, keepdim=True)
                scale = (self.weight - mean).abs().mean(0, keepdim=True)
                self.wscale.data = scale
                self.wbias.data = mean
        elif self.scaling_pattern == 'single':
            self.wscale = nn.Parameter(torch.zeros(1), requires_grad=True)
            self.wbias = nn.Parameter(torch.zeros(1), requires_grad=True)
            if init_method == 'xnor':
                mean = self.weight.mean()
                scale = (self.weight - mean).abs().mean()
                self.wscale.data = scale
                self.wbias.data = mean


    def forward(self, x):
        w = STEBinary().apply(self.weight)
        w = self.wscale * w + self.wbias
        out = F.linear(x, w, self.bias)
        return out
    
    def binarize(self):
        binary_weight = STEBinary().apply(self.weight)
        w = binary_weight * self.wscale.data + self.wbias.data
        return w
    
    def to_regular_linear(self):
        w = self.binarize()
        linear = nn.Linear(w.shape[1], w.shape[0], bias=self.bias is not None)
        linear.weight.data = w
        if self.bias is not None:
            linear.bias.data = self.bias
        return linear