import torch
import torch.nn as nn
import copy
from Quantization_function import *
from Quantization_module import *


def quantize_model(model):
    """
    Recursively quantize a pretrained single-precision model to int8 quantized model
    model: pretrained single-precision model
    """
    # quantize convolutional and linear layers to 8-bit
    if type(model) == nn.Conv2d: # if encounter a 'nn.Conv2d' layer
        quant_mod = Quant_Conv2d(weight_bit = 2)
        quant_mod.set_param(model) # set the Quant_Conv2d with the parameter from the original Conv2d layer
        return quant_mod # replace the original Conv2d layer to the new Quant_Conv2d
    elif type(model) == nn.Linear: # same comment as above 
        quant_mod = Quant_Linear(weight_bit = 2)
        quant_mod.set_param(model)
        return quant_mod

    # quantize all the activation to 8-bit
    elif type(model) == nn.ReLU or type(model) == nn.ReLU6:
        return nn.Sequential(*[model, QuantAct(activation_bit=2)]) # return all the NN layers and add a QuantAct layer with 8 bits

    # recursively use the quantized module to replace the single-precision module
    elif type(model) == nn.Sequential:
        mods = []
        for m in model.children(): # m stands for the layers inside the nn.Sequential()
            mods.append(quantize_model(m)) # quantize everything inside nn.Sequential()
        return nn.Sequential(*mods) # * will unpack the list mods and passing its element as separate arguments to the 'nn.Sequential' constructor
    else:
        q_model = copy.deepcopy(model) # deep copy ensures that the original model remains unchanged during the quantization process
        for attr in dir(model):
            mod = getattr(model, attr) # get attribute in model 
            if isinstance(mod, nn.Module) and 'norm' not in attr: # ensure there is no normalization layer is being quantized
                setattr(q_model, attr, quantize_model(mod)) # the attribute inside q_model will be replaced by the quantized layer.
        return q_model
    
def freeze_model(model):
    """
    freeze the activation range
    """
    if type(model) == QuantAct: # Assume when encounter the first QuantAct, the x_max and x_min is optimal
        model.fix() #works the same as above
    elif type(model) == nn.Sequential:
        for m in model.children():
            freeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                freeze_model(mod)
        return model

def unfreeze_model(model): # works the same as 'freeze_model' but unfreeze the model for this one
    """
    unfreeze the activation range
    """
    if type(model) == QuantAct:
        model.unfix()
    elif type(model) == nn.Sequential:
        for m in model.children():
            unfreeze_model(m)
    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                unfreeze_model(mod)
        return model