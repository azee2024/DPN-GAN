# Import layer
import numpy as np
import torch
from torch import nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from deconv1d.nn import DeformConv1d


class DeformConvCustom(nn.Module):
    def __init__(self, out_channels, kernel_size):
        super(DeformConvCustom, self).__init__()
        self.device = 'cpu'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        kh = kw = self.kernel_size
        mel_channels = x.shape[1]
        mel_timeframes = x.shape[3]
        padding = self.kernel_size // 2

        pd = (padding, padding, padding, padding)  # pad last dim by 1 on each side
        x = F.pad(x, pd, "constant", 0)  # effectively zero padding

        offset_layer = nn.Conv2d(in_channels, 2 * kh * kw, kernel_size=self.kernel_size)
        offsets = offset_layer(x)

        # Output shape
        out_filters = x.shape[2] - kh + 1
        out_timeframes = x.shape[3] - kw + 1

        weight = torch.rand(self.out_channels, mel_channels, kh, kw)
        mask = torch.rand(batch_size, kh * kw, out_filters, out_timeframes)

        x = deform_conv2d(x, offsets, weight, mask=mask)
        x = F.layer_norm(x, normalized_shape=x.shape)

        # Activation
        act = PRAK()
        x = act(x)

        return x


class DPNResBlock(nn.Module):
    def __init__(self, out_channels, kernel_size):
        super(DPNResBlock, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.device = 'cpu'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        batch_size = x.shape[0]
        in_channels = x.shape[1]
        kh = kw = self.kernel_size
        mel_channels = x.shape[1]
        mel_timeframes = x.shape[3]
        padding = self.kernel_size // 2

        x_addon = x.detach()

        pd = (padding, padding, padding, padding)  # pad last dim by 1 on each side
        x = F.pad(x, pd, "constant", 0)  # effectively zero padding

        offset_layer = nn.Conv2d(in_channels, 2 * kh * kw, kernel_size=self.kernel_size)
        offsets = offset_layer(x)

        # Output shape
        out_filters = x.shape[2] - kh + 1
        out_timeframes = x.shape[3] - kw + 1

        weight = torch.rand(self.out_channels, mel_channels, kh, kw)
        mask = torch.rand(batch_size, kh * kw, out_filters, out_timeframes)

        x = deform_conv2d(x, offsets, weight, mask=mask)
        x = F.layer_norm(x, normalized_shape=x.shape)

        # Activation
        act = PRAK()
        x = act(x)

        x = torch.cat([x, x_addon], dim=1)

        pool_layer = nn.MaxPool2d(kernel_size=self.kernel_size, stride=2)
        x = pool_layer(x)

        return x


# Generator Utilities
class DeConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(DeConv1d, self).__init__()
        self.device = 'cpu'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.deconv = DeformConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )

    def forward(self, x, offsets):
        return self.deconv(x, offsets)


class PRAK(nn.Module):
    def __init__(self, preluconst=1.0):
        super(PRAK, self).__init__()
        self.preluconst = preluconst

    def forward(self, x):
        return self.preluconst * (self._trianglewave(x) + self._trianglewave(x + np.pi / 2))

    def _trianglewave(self, x):
        return (x - np.pi * torch.floor(x / np.pi + 0.5)) * (-1) ** torch.floor(x / np.pi + 0.5)


class DPNBlock(nn.Module):
    def __init__(self, params):
        super(DPNBlock, self).__init__()
        self.params = params

        # Extracting parameters
        self.in_channels = params["in_channels"]
        self.out_channels = params["out_channels"]
        self.kernel_size = params["kernel_size"]
        self.input_shape = params["shape"]
        self.lstm_num_layers = params["lstm_num_layers"]
        self.dpn_depth = params["dpn_depth"]

    def forward(self, x):
        dpn_resblocks = []
        for i in range(self.dpn_depth):
            layer = DPNResBlock(out_channels=self.out_channels * (i + 1), kernel_size=self.kernel_size)
            dpn_resblocks.append(layer)

        for block in dpn_resblocks:
            x = block(x)

        return x


class MelInitiatorBlock(nn.Module):
    def __init__(self, params):
        super(MelInitiatorBlock, self).__init__()

        # Extracting parameters
        self.in_channels = params["in_channels"]
        self.out_channels = params["out_channels"]
        self.kernel_size = params["kernel_size"]
        # self.lstm_hidden_size = params["lstm_hidden_size"]
        self.lstm_num_layers = params["lstm_num_layers"]
        self.input_shape = params["input_shape"]

        # Deformable Conv 1
        self.deformconv1 = DeformConvCustom(out_channels=self.out_channels // 2, kernel_size=self.kernel_size)
        self.deformconv2 = DeformConvCustom(out_channels=self.out_channels, kernel_size=self.kernel_size)

    def forward(self, x):
        # Shape of X should be (batch_size, mel_channels, mel_filters, mel_timeframe)

        # Deformable convolution
        x = self.deformconv1(x)
        x = self.deformconv2(x)
        return x


class MetaInitiatorBlock(nn.Module):
    def __init__(self, params):
        super(MetaInitiatorBlock, self).__init__()
        self.params = params
        self.device = "cpu"  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_size = self.params["input_size"]
        hidden_size = self.params["hidden_size"]
        output_size = self.params["output_size"]

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Flatten the input tensor if it's not already flattened
        # if len(x.shape) > 2:
        #     x = x.view(x.size(0), -1)

        # Pass through the first fully connected layer
        x = torch.squeeze(x, dim=1)
        x = self.fc1(x)
        x = torch.relu(x)

        # Pass through the second fully connected layer
        x = self.fc2(x)

        # Add a new dimension to the output tensor
        x = torch.unsqueeze(x, 1)
        x = x.transpose(1, 2)
        x = torch.unsqueeze(x, 2)

        return x


class MelMetaConcatenator(nn.Module):
    def __init__(self):
        super(MelMetaConcatenator, self).__init__()

    def forward(self, x1, x2):
        # Expand tensor2 to match the shape of tensor1
        x2 = x2.expand(-1, -1, x1.shape[2], x1.shape[3])

        # Concatenate along a new dimension, for example along the third dimension (dim=2)
        concatenated = x1 * x2
        return concatenated


class Modulator(nn.Module):
    def __init__(self, params):
        super(Modulator, self).__init__()
        self.params = params
        self.device = 'cpu'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.convt_in_channels = self.params["ConvT_in_channels"]
        self.convt_out_channels = self.params["ConvT_out_channels"]
        self.convt_kernel_size = self.params["ConvT_kernel_size"]

        self.convt = nn.ConvTranspose2d(in_channels=self.convt_in_channels,
                                        out_channels=self.convt_out_channels,
                                        kernel_size=self.convt_kernel_size,
                                        stride=2)
        self.dpn = DPNBlock(params=params)

    def forward(self, x):
        x = self.convt(x)
        x = self.dpn(x)
        return x


class ProcessorBlock(nn.Module):
    def __init__(self, params):
        super(ProcessorBlock, self).__init__()
        self.params = params
        self.device = 'cpu'  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Extracting parameters
        self.in_channels = params["in_channels"]
        self.out_channels = params["out_channels"]
        self.kernel_size = params["kernel_size"]
        self.input_shape = params["shape"]
        self.out_seq_length = params["out_seq_length"]
        self.kh = self.kernel_size
        self.kw = self.kernel_size

        # Deformable Convolution
        self.deformconv = DeformConvCustom(out_channels=self.out_channels, kernel_size=self.kernel_size)

        # MaxPool Layer for Downsample
        self.pool1 = nn.MaxPool2d(kernel_size=self.kernel_size, stride=2)

        # PRAK Activation
        self.dropout = nn.Dropout(p=0.4)
        self.prak = PRAK()

        # Point Convolution
        self.point_conv = None

        # PRAK 2
        self.prak2 = PRAK()

        # Output Layer
        self.dense2 = None
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Deformable convolution 1
        x = self.deformconv(x)

        # Maxpooling 1
        x = self.pool1(x)

        # Activation
        x = self.dropout(x)
        x = self.prak(x)

        # Point Convolution
        self.point_conv = nn.Conv2d(in_channels=x.shape[1], out_channels=1, kernel_size=1, stride=1)
        x = self.point_conv(x)
        x = self.prak2(x)

        # Flattening
        x = x.view(x.size(0), 1, -1)

        self.dense2 = nn.Linear(in_features=x.shape[-1], out_features=self.out_seq_length)
        x = self.dense2(x)
        x = self.tanh(x)

        return x
