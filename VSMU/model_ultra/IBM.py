import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PRM(nn.Module):
    def __init__(self, img_size=240, kernel_size=3, downsample_ratio=1, dilations=[1, 6, 12], embed_dim=64,
                 share_weights=False, op='cat'):
        super().__init__()
        self.dilations = dilations
        self.embed_dim = embed_dim
        self.downsample_ratio = downsample_ratio
        self.op = op
        self.kernel_size = kernel_size
        self.stride = downsample_ratio
        self.share_weights = share_weights

        # 调整卷积层的输入通道数
        self.channel = nn.Conv3d(in_channels=embed_dim * len(dilations), out_channels=embed_dim, kernel_size=1)

        if share_weights:
            self.convolution = nn.Conv3d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=self.kernel_size,
                                         stride=1, padding=(self.kernel_size - 1) * dilations[0] // 2, dilation=dilations[0])
        else:
            self.convs = nn.ModuleList()
            for dilation in self.dilations:
                padding = (self.kernel_size - 1) * dilation // 2
                self.convs.append(nn.Sequential(
                    nn.Conv3d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=self.kernel_size,
                              stride=self.stride, padding=padding, dilation=dilation),
                    nn.GELU()))

        if self.op == 'sum':
            self.out_chans = embed_dim
        elif self.op == 'cat':
            self.out_chans = embed_dim * len(self.dilations)

    def forward(self, x):
        if self.share_weights:
            padding = (self.kernel_size - 1) * self.dilations[0] // 2
            y = F.conv3d(x, weight=self.convolution.weight, bias=self.convolution.bias,
                         stride=self.downsample_ratio, padding=padding,
                         dilation=self.dilations[0]).unsqueeze(dim=-1)
            for i in range(1, len(self.dilations)):
                padding = (self.kernel_size - 1) * self.dilations[i] // 2
                _y = F.conv3d(x, weight=self.convolution.weight, bias=self.convolution.bias,
                              stride=self.downsample_ratio, padding=padding,
                              dilation=self.dilations[i]).unsqueeze(dim=-1)
                y = torch.cat((y, _y), dim=-1)
        else:
            y = self.convs[0](x).unsqueeze(dim=-1)
            for i in range(1, len(self.dilations)):
                _y = self.convs[i](x).unsqueeze(dim=-1)
                y = torch.cat((y, _y), dim=-1)

        B, C, W, H, D, N = y.shape

        if self.op == 'sum':
            y = y.sum(dim=-1).flatten(2).permute(0, 2, 1).contiguous()
        elif self.op == 'cat':
            y = y.permute(0, 5, 1, 2, 3, 4).flatten(2).reshape(B, N * C, W, H, D).contiguous()
            y_c = self.channel(y)
        else:
            raise NotImplementedError('no such operation: {} for multi-levels!'.format(self.op))

        return y_c

