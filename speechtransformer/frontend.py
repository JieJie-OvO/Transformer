import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def cal_width_dim_2d(input_dim, kernel_size, stride, padding=1):
    return math.floor((input_dim + 2 * padding - kernel_size)/stride + 1)


class Conv2dLayer(nn.Module):
    def __init__(self, input_size, in_channel, out_channel, kernel_size, stride,
                 dropout=0.1):
        super(Conv2dLayer, self).__init__()

        self.input_size = input_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (0, kernel_size // 2)

        self.conv_layer = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding)

        self.output_size = cal_width_dim_2d(input_size, self.kernel_size, self.stride, padding=self.padding[1])


        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        out = self.conv_layer(x)
        out = F.relu(out)
        out = self.dropout(out)
        mask = self.return_output_mask(mask, out.size(2))
        return out, mask

    def return_output_mask(self, mask, t):
        # conv1
        stride = self.stride
        kernel_size = self.kernel_size
        mask = mask[:, math.floor(kernel_size / 2)::stride][:,:t]
        return mask


class ConvFrontEnd(nn.Module):
    def __init__(self, input_size=40, output_size=256, channel=[1,64,128], 
                 kernel_size=[3,3], stride=[2, 2], dropout=0.0):
        super(ConvFrontEnd, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.output_size = output_size

        self.conv1 = Conv2dLayer(
            input_size=input_size,
            in_channel=channel[0],
            out_channel=channel[1],
            kernel_size=self.kernel_size[0],
            stride=self.stride[0],
            dropout=dropout)

        self.conv2 = Conv2dLayer(
            self.conv1.output_size,
            in_channel=channel[1],
            out_channel=channel[2],
            kernel_size=self.kernel_size[1],
            stride=self.stride[1],
            dropout=dropout)

        self.conv_output_size = self.conv2.output_size * self.conv2.out_channel
        self.output_layer = nn.Linear(self.conv_output_size, self.output_size)

    def forward(self, x, mask):
        x = x.unsqueeze(1)
        x, mask = self.conv1(x, mask)
        x, mask = self.conv2(x, mask)
        
        b, c, t, f = x.size()
        x = x.transpose(1, 2).reshape(b, t, c * f)
        x = self.output_layer(x)

        return x, mask
