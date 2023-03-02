import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm
from utils import init_weights, get_padding
import numpy as np

LRELU_SLOPE = 0.1

class ResBlock(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

class NSPP_Model(torch.nn.Module):
    def __init__(self, h):
        super(NSPP_Model, self).__init__()
        self.h = h
        self.num_kernels = len(h.ResNet_ResBlock_kernel_sizes)

        self.input_conv = weight_norm(Conv1d(h.n_fft//2+1, h.ResNet_channel, h.ResNet_input_conv_kernel_size, 1, 
                                             padding=get_padding(h.ResNet_input_conv_kernel_size, 1)))

        self.ResNet = nn.ModuleList()
        for j, (k, d) in enumerate(zip(h.ResNet_ResBlock_kernel_sizes, h.ResNet_ResBlock_dilation_sizes)):
            self.ResNet.append(ResBlock(h.ResNet_channel, k, d))

        self.PEA_conv_R = weight_norm(Conv1d(h.ResNet_channel, h.n_fft//2+1, h.PEA_R_conv_kernel_size, 1, 
                                             padding=get_padding(h.PEA_R_conv_kernel_size, 1)))
        self.PEA_conv_I = weight_norm(Conv1d(h.ResNet_channel, h.n_fft//2+1, h.PEA_I_conv_kernel_size, 1, 
                                             padding=get_padding(h.PEA_I_conv_kernel_size, 1)))

        self.PEA_conv_R.apply(init_weights)
        self.PEA_conv_I.apply(init_weights)

    def forward(self, log_amp):

        output = self.input_conv(log_amp)
        outputs = None
        for j in range(self.num_kernels):
            if outputs is None:
                outputs = self.ResNet[j](output)
            else:
                outputs += self.ResNet[j](output)
        output = outputs / self.num_kernels
        output = F.leaky_relu(output)   

        pseudo_real_part = self.PEA_conv_R(output)
        pseudo_imaginary_part = self.PEA_conv_I(output)

        phase = torch.atan2(pseudo_imaginary_part, pseudo_real_part)

        return phase

def losses(phase_r, phase_g, n_fft, frames):

    IP_r = phase_r
    IP_g = phase_g

    GD_matrix = torch.triu(torch.ones(n_fft//2+1, n_fft//2+1), diagonal=1) - torch.triu(torch.ones(n_fft//2+1, n_fft//2+1), diagonal=2) - torch.eye(n_fft//2 + 1)
    GD_matrix = GD_matrix.to(phase_g.device)

    GD_r = torch.matmul(phase_r.permute(0,2,1), GD_matrix)
    GD_g = torch.matmul(phase_g.permute(0,2,1), GD_matrix)

    IAF_matrix = torch.triu(torch.ones(frames, frames), diagonal=1) - torch.triu(torch.ones(frames, frames), diagonal=2) - torch.eye(frames)
    IAF_matrix = IAF_matrix.to(phase_g.device)

    IAF_r = torch.matmul(phase_r, IAF_matrix)
    IAF_g = torch.matmul(phase_g, IAF_matrix)

    L_IP = torch.mean(anti_wrapping_function(IP_r-IP_g))
    L_GD = torch.mean(anti_wrapping_function(GD_r-GD_g))
    L_IAF = torch.mean(anti_wrapping_function(IAF_r-IAF_g))

    return L_IP, L_GD, L_IAF

def anti_wrapping_function(x):

    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)
