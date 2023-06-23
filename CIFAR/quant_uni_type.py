import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


bit_alpha = {1:1.2221, 2:0.6511, 4:0.1946}

class QuantReLUFunction(torch.autograd.Function):
    def __init__(self, bit = 1, ffun = 'quant', bfun = 'inplt', rate_factor = 0.01, gd_alpha = False, gd_type = 'mean'):
        QuantReLUFunction.rate_factor   = rate_factor
        QuantReLUFunction.ffun          = ffun
        QuantReLUFunction.bfun          = bfun
        QuantReLUFunction.bit           = bit
        QuantReLUFunction.gd_alpha      = gd_alpha
        QuantReLUFunction.gd_type       = gd_type
    @staticmethod
    def forward(ctx, input, alpha):#, rate_factor=0.01, ffun='quant', bfun='inplt', bit=4, gd_alpha=True, gd_type='mean'):
        alpha.data.clamp_(min=1e-3, max=5)
        a = alpha.item()
        level = 2 ** QuantReLUFunction.bit - 1
        ctx.save_for_backward(input, alpha)
        output = input.clone()
        output[input <= 0] = 0
        if QuantReLUFunction.ffun == 'quant':
            ind = input.gt(0) * input.le(a*(level-1))
            output[ind] = (input[ind] / a).ceil() * a
            output[input > a*(level-1)] = a * level
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, alpha = ctx.saved_tensors
        a = alpha.item()
        level = 2 ** QuantReLUFunction.bit - 1
        grad_input = grad_alpha = None
        grad_input = grad_output.clone()
        if QuantReLUFunction.bfun != 'idnty':
            grad_input[input <= 0] = 0
        if QuantReLUFunction.bfun == 'inplt':
            grad_input[input > a*level] = 0
        if QuantReLUFunction.gd_alpha and QuantReLUFunction.ffun == 'quant':
            grad_0 = torch.zeros_like(grad_output)
            if QuantReLUFunction.gd_type == 'ae':
                ind = input.gt(0) * input.le(a*(level-1))
                grad_0[ind] = (input[ind]/a).ceil()
                grad_0[input > a*(level-1)] = level
            elif QuantReLUFunction.gd_type == 'pact':
                grad_0[input > a*level] = level
            elif QuantReLUFunction.gd_type == 'min':
                grad_0[input.gt(0) & input.le(a*level)] = 1
            elif QuantReLUFunction.gd_type == 'mean':
                grad_0[input.gt(0) & input.le(a*level)] = 2 ** (QuantReLUFunction.bit - 1)
            else:
                raise NotImplementedError
            # grad_0[ind] = 1
            grad_0[input > a*level] = level
            grad_alpha = (grad_output * grad_0).sum() * QuantReLUFunction.rate_factor
            grad_alpha = torch.clamp(grad_alpha,-1,1)
            grad_alpha = grad_alpha.expand(1)
        elif not QuantReLUFunction.ffun == 'quant':
            alpha.data.fill_(input.abs().max() / level)
        return grad_input, grad_alpha

class QuantReLU(nn.Module):
    def __init__(self, bit, ffun = 'relu', bfun = 'relu', rate_factor = 0.01, gd_alpha = False, gd_type = 'mean'):
        super(QuantReLU, self).__init__()
        self.fw = ffun
        self.bw = bfun
        self.bit = bit
        self.a = bit_alpha[bit]
        self.r = rate_factor
        self.g = int(gd_alpha==True)
        self.alpha = nn.Parameter(torch.Tensor(1))
        self.rate_factor = nn.Parameter(torch.Tensor(1), requires_grad=False)
        self.gd_alpha = nn.Parameter(torch.Tensor(1), requires_grad=False)
        self.g_t = gd_type
        self.reset_parameters()
        
    def reset_parameters(self):
        self.alpha.data = torch.ones(1) * self.a
        self.rate_factor.data = torch.ones(1) * self.r
        self.gd_alpha.data = torch.ones(1) * self.g
        
    def forward(self, input):
        r = self.rate_factor.item()
        g = bool(self.gd_alpha.item())
        qtuni = QuantReLUFunction(self.bit,ffun=self.fw,bfun=self.bw,rate_factor=r,gd_alpha=g,gd_type=self.g_t)
        return qtuni.apply(input, self.alpha)
