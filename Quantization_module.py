import torch
import time
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
import sys
from Quantization_function import *

class QuantAct(nn.Module): # QuantAct is designed for quantizing the activation layer of the NNs 
    def __init__(self, activation_bit, full_precision_flag=False, running_stat=True):
        super(QuantAct, self).__init__()
        self.activation_bit = activation_bit # activation_bit is the bit-setting after quantization applied, typically 8 bit
        self.momentum = 0.99   
        self.full_precision_flag = full_precision_flag
        self.running_stat = running_stat
        self.register_buffer('x_min', torch.zeros(1)) #register_buffer they do not get updated with gradient descent during the training process.
        self.register_buffer('x_max', torch.zeros(1)) # Initialized to a tensor containing a single zero
        self.act_function = AsymmetricQuantFunction.apply # Used to implment custom 'forward' and 'backward' method

    def __repr__(self): # define the "offical" string representation of an object
        return "{0}(activation_bit={1}, full_precision_flag={2}, running_stat={3}, Act_min: {4:.2f}, Act_max: {5:.2f})".format(
            self.__class__.__name__, self.activation_bit,
            self.full_precision_flag, self.running_stat, self.x_min.item(),
            self.x_max.item()) # E.g. QuantAct(activation_bit=8, full_precision_flag=False, running_stat=True, Act_min: 0.00, Act_max: 1.00)

    def fix(self): # After enough number of epochs, the x_min and x_max have stabilized and ready to get fixed
        self.running_stat = False

    def forward(self, x):
        if self.running_stat:
            x_min = x.detach().min() # modify from .data to .detach
            x_max = x.detach().max() # Any operations performed on this detached tensor will not be recorded for gradient computation during backpropagation
            self.x_min = min(self.x_min, x_min) # modify from self.x_min += -self.x_min + min(self.x_min, x_min)
            self.x_max = max(self.x_max, x_max) # update the class attributes

        if not self.full_precision_flag: # when full_precision_flag = False, return the quantized result of input
            quant_act = self.act_function(x, self.activation_bit, self.x_min, self.x_max) 
            return quant_act
        else:
            return x
        
class Quant_Linear(nn.Module): # Quant_Linear is designed for quantizing the Linear layer weights of the NNs
    def __init__(self, weight_bit, full_precision_flag=False):
        super(Quant_Linear, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit # weight_bit is the bit-setting after quantization applied, typically 8 bit
        self.weight_function = AsymmetricQuantFunction.apply # Used to implment custom 'forward' and 'backward' method

    def __repr__(self): # define the "offical" string representation of an object
        base_repr = super(Quant_Linear, self).__repr__()
        return "{}(base={}, weight_bit={}, full_precision_flag={})".format(self.__class__.__name__, base_repr, self.weight_bit, self.full_precision_flag)
        # E.g. Quant_Linear(base=Module(), weight_bit=8, full_precision_flag=False)

    def set_param(self, linear):
        self.in_features = linear.in_features # store the in_features in class attribute
        self.out_features = linear.out_features # store the out_features in class attribute
        self.weight = nn.Parameter(linear.weight.detach().clone()) 
        # .clone() create an independent copy of the tensor
        # using nn.Parameter can collect all parameters of a model for gradient computation and for updating the parameters during training with an optimizer.
        try:
            self.bias = nn.Parameter(linear.bias.detach().clone())
        except AttributeError: # since linear layer can be created without bias
            self.bias = None   # E.g. linear = nn.Linear(in_features=10, out_features=5, bias=False)

    def forward(self, x):
        w = self.weight
        x_transform = w.detach() 
        w_min = x_transform.min(dim = 1).values # if dim = 1, Find the maximum value in each row across all columns. so here find the max and min for each output neuron
        w_max = x_transform.max(dim = 1).values # .min() and .max() will return 'values' and 'indices', .values will extract the tensor of minimum or maximum values
        if not self.full_precision_flag: # when full_precision_flag = False, return the quantized result of input weight
            w = self.weight_function(self.weight, self.weight_bit, w_min, w_max)
        else:
            w = self.weight
        return F.linear(x, weight=w, bias=self.bias) # return the new output = new_w * input + bias       
    
class Quant_Conv2d(nn.Module): # Quant_Linear is designed for quantizing the convolutional layer weights of the NNs
    def __init__(self, weight_bit, full_precision_flag=False):
        super(Quant_Conv2d, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit # weight_bit is the bit-setting after quantization applied, typically 8 bit
        self.weight_function = AsymmetricQuantFunction.apply # Used to implment custom 'forward' and 'backward' method

    def __repr__(self): # define the "offical" string representation of an object
        base_repr = super(Quant_Conv2d, self).__repr__()
        return "{}(base={}, weight_bit={}, full_precision_flag={})".format(self.__class__.__name__, base_repr, self.weight_bit, self.full_precision_flag) 
        # E.g. Quant_Conv2d(base=Module(), weight_bit=8, full_precision_flag=False)

    def set_param(self, conv):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.weight = nn.Parameter(conv.weight.detach().clone())
        try:
            self.bias = nn.Parameter(conv.bias.detach().clone())
        except AttributeError:
            self.bias = None

    def forward(self, x):
        w = self.weight
        x_transform = w.detach().contiguous().view(self.out_channels, -1) # flatten the weight to row of out_channels and column with enough data space
        w_min = x_transform.min(dim = 1).values #if dim = 0, Find the maximum value in each column across all rows.
        w_max = x_transform.max(dim = 1).values #if dim = 1, Find the maximum value in each row across all columns. so here find the max and min for each channel
        if not self.full_precision_flag:
            w = self.weight_function(self.weight, self.weight_bit, w_min, w_max)
        else:
            w = self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)