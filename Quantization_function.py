import math
import numpy as np
from torch.autograd import Function
import torch

# for Linear_quantize equation:
# Fix point (floating) number = (Quantized fix-point integer + Zero_point) / Scale
# Quantized fix-point integer = Scale * Floating point real number - Zero_point


def clamp(input, min, max, inplace = False): # A tensor with all element clamped to the [min,max] range.
    # if 'inplace = true', then performs the operation in-place and modifies the input tensor
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max) #if 'inplace = false', then return a new tensor.

def linear_quantize(input, scale, zero_point, inplace=False): # q = Scale * r - Zero-point
    # if the input are CNNs, then reshape the scale & zero-point to 4D tensor 
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
        # -1 argument will return the same size as the original scale
        # 1,1,1 one channel with height one and width one

    # if the input are Linear inputs, then reshape the scale & zero-point to 2D tensor
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
        # -1 argument will return the same row size as the original scale
        # 1 argument will return only one column

    # mapping single-precision input(float32) to integer values with the given scale and zeropoint
    # if 'inplace = true', then performs the operation in-place and modifies the input tensor
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    return torch.round(scale * input - zero_point) #if 'inplace = false', then return a new tensor.

def linear_dequantize(input, scale, zero_point, inplace=False): # r = (q + Zero-point) / Scale
    # if the input are CNNs, then reshape the scale & zero-point to 4D tensor
    if len(input.shape) == 4:
        scale = scale.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
        # -1 argument will return the same size as the original scale
        # 1,1,1 one channel with height one and width one

    # if the input are Linear inputs, then reshape the scale & zero-point to 2D tensor
    elif len(input.shape) == 2:
        scale = scale.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
        # -1 argument will return the same row size as the original scale
        # 1 argument will return only one column

    # mapping integer input to fixed point float point value with given scaling factor and zeropoint
    # if 'inplace = true', then performs the operation in-place and modifies the input tensor
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    return (input + zero_point) / scale #if 'inplace = false', then return a new tensor.

def asymmetric_linear_quantization_params(num_bits, saturation_min, saturation_max, integral_zero_point=True, signed=True):
    # Computes the scaling factor and zero point for quantization, given the number of bits and the range of values (saturation points).
    # saturation_min: r_min
    # saturation_max: r_max
    n = 2**num_bits - 1 # n is q_max
    scale = n / torch.clamp((saturation_max - saturation_min), min = 1e-8) # S = (q_max - q_min) / (r_max - r_min) , q_min = 0
    zero_point = scale * saturation_min # Z = S * r_min - q_min , q_min = 0
    if integral_zero_point: # If a integral zero point is needed then 'integral_zero_point = True'
        if isinstance(zero_point, torch.Tensor): # If zero_point is a torch.Tensor then perform a round operation
            zero_point = zero_point.round()
        else: #If zero_point is not a torch.Tensor then perform a round operation and convert it to 'float' type number
            zero_point = float(round(zero_point))
    if signed:
        zero_point += 2**(num_bits - 1)
    return scale, zero_point    

class AsymmetricQuantFunction(Function):
    def forward(ctx, x, k, x_min=None, x_max=None):
        # If no x_min and x_max input then find the x_min and x_max from the input.
        if x_min is None or x_max is None: # or (sum(x_min == x_max) == 1 and x_min.numel() == 1): [considered as useless now]
            x_min, x_max = x.min(), x.max()
        scale, zero_point = asymmetric_linear_quantization_params(k, x_min, x_max) # return scale and zeropint
        new_quant_x = linear_quantize(x, scale, zero_point, inplace=False) # return q
        n = 2**(k - 1)
        new_quant_x = torch.clamp(new_quant_x, -n, n - 1) # Ensure the q is in the range of quantized signed setting
                                                          # i.e k = 8, then new_quant_x will be clamped in the range of -128 to 127  
        quant_x = linear_dequantize(new_quant_x, scale, zero_point, inplace=False) # return r
        return quant_x