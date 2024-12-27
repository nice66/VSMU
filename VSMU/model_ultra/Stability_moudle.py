import torch
from torch import nn
import torch.nn.functional as F
import numpy as np



class EinFFT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden_size = dim
        self.num_blocks = 4
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        self.complex_weight_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_weight_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_bias_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_bias_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * self.scale)

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):
        B, H, W, L, C = x.shape
        N = H * W * L
        x = x.view(B, N, self.num_blocks, self.block_size)

        x = torch.fft.fft2(x, dim=(1, 2), norm='ortho')  # FFT on N dimension

        x_real_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[0]) - self.multiply(x.imag, self.complex_weight_1[1]) +
            self.complex_bias_1[0])
        x_imag_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[1]) + self.multiply(x.imag, self.complex_weight_1[0]) +
            self.complex_bias_1[1])
        x_real_2 = self.multiply(x_real_1, self.complex_weight_2[0]) - self.multiply(x_imag_1,
                                                                                     self.complex_weight_2[1]) + \
                   self.complex_bias_2[0]
        x_imag_2 = self.multiply(x_real_1, self.complex_weight_2[1]) + self.multiply(x_imag_1,
                                                                                     self.complex_weight_2[0]) + \
                   self.complex_bias_2[1]

        x = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        x = F.softshrink(x, lambd=self.sparsity_threshold) if self.sparsity_threshold else x
        x = torch.view_as_complex(x)

        x = torch.fft.ifft2(x, dim=(1, 2), norm="ortho")

        # RuntimeError: "fused_dropout" not implemented for 'ComplexFloat'
        x_real = x.real.to(torch.float32)
        x = x_real
        # x = x.to(torch.float32)
        x = x.reshape(B, C, H, W, L)
        return x