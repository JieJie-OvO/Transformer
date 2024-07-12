import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class Swish(nn.Module):
    def __init__(self, factor = 1.0):
        super(Swish, self).__init__()
        self.factor = factor

    def forward(self, x):
        return x * torch.sigmoid(x * self.factor)


class GLU(nn.Module):
    def __init__(self, dim: int) -> None:
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, x):
        outputs, gate = x.chunk(2, dim=self.dim)
        return outputs * gate.sigmoid()

class DepthwiseConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
            stride = 1, padding = -1, bias=False):
        super(DepthwiseConv1d, self).__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        
        padding = (kernel_size - 1) // 2 if padding == -1 else padding

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)


class PointwiseConv1d(nn.Module):
    def __init__(self, in_channels,  out_channels, stride = 1,
            padding = 0, bias = True):
        
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)


class ConformerConvModule(nn.Module):
    def __init__(
            self, in_channels, kernel_size=7,
            expansion_factor = 2, dropout_p = 0.1):
        super(ConformerConvModule, self).__init__()

        self.norm = nn.LayerNorm(in_channels)
        self.pointwise_conv1 = PointwiseConv1d(in_channels, in_channels * expansion_factor)
        self.glu = GLU(dim=1)

        self.depthwise_conv = DepthwiseConv1d(in_channels, in_channels, kernel_size)
        self.batch_norm = nn.BatchNorm1d(in_channels)
        self.swish = Swish()

        self.pointwise_conv2 = PointwiseConv1d(in_channels, in_channels)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, mask):
        mask = mask.unsqueeze(2).repeat([1, 1, x.size(-1)])

        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        x = x.transpose(1, 2)
        x.masked_fill_(~mask, 0.0)

        x = x.transpose(1, 2)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)

        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        x.masked_fill_(~mask, 0.0)

        return x