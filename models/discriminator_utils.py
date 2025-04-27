import torch
from torch import nn as nn
import torch.nn.functional as F

from models.generator_utils import DeConv1d, PRAK, DPNResBlock


class MSDRescalingLayer(nn.Module):
    def __init__(self, out_channels, kernel_size, padding, stride):
        super(MSDRescalingLayer, self).__init__()
        self.layer_norm = None
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        x_addon = x.detach()
        offsets = torch.randn(x.shape[0], 1, x.shape[-1] + 2 * self.padding - self.kernel_size + 1, self.kernel_size)
        defconv1d = DeConv1d(in_channels=x.shape[1],
                             out_channels=self.out_channels,
                             kernel_size=self.kernel_size,
                             padding=self.padding,
                             stride=self.stride)
        x = defconv1d(x, offsets)

        self.layer_norm = nn.LayerNorm(normalized_shape=x.shape)
        x = self.layer_norm(x)

        x = torch.cat([x, x_addon], dim=1)
        pool1 = nn.MaxPool1d(kernel_size=self.kernel_size,
                             stride=self.stride,
                             padding=self.padding)
        x = pool1(x)

        prak1 = PRAK()
        x = prak1(x)

        # x = torch.cat([x, x_addon], dim=1)
        return x


# Discriminator Utilities
class MSDInitiator(nn.Module):
    def __init__(self, params):
        super(MSDInitiator, self).__init__()
        self.params = params
        self.kernel_size = params["kernel_size"]
        self.stride = params["stride"]
        self.avg_pooler = nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        x = self.avg_pooler(x)
        return x


class Distributor(nn.Module):
    def __init__(self, params):
        super(Distributor, self).__init__()
        self.params = params
        self.depth = params["depth"]

        self.defconv_in_channels = params["defconv_in_channels"]
        self.defconv_out_channels = params["defconv_out_channels"]
        self.kernel_size = self.params["kernel_size"]
        self.stride = params["stride"]
        self.padding = (self.kernel_size - 1) // 2
        self.kh = self.defconv_in_channels

    def forward(self, x):
        for _ in range(self.depth):
            rescale = MSDRescalingLayer(out_channels=self.defconv_out_channels,
                                        kernel_size=self.kernel_size,
                                        padding=self.padding,
                                        stride=self.stride)
            x = rescale(x)

            pool = nn.MaxPool1d(kernel_size=self.params["kernel_size"], stride=4)
            x = pool(x)

            self.params["defconv_in_channels"] = x.shape[1]
            self.params["defconv_out_channels"] = x.shape[1] * 2
        return x


class MSDFinal(nn.Module):
    def __init__(self, params):
        super(MSDFinal, self).__init__()
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.kernel_size = self.params["kernel_size"]
        self.stride = self.params["stride"]
        self.hidden_dims = self.params["hidden_dims"]

        self.flatten = nn.Flatten()
        self.avgpool1d = nn.AvgPool1d(kernel_size=self.kernel_size, stride=self.stride)

        self.dense1 = None
        self.act1 = nn.ReLU()
        self.dense2 = nn.Linear(in_features=self.hidden_dims * 4, out_features=self.hidden_dims)
        self.act2 = nn.ReLU()
        self.dense3 = nn.Linear(in_features=self.hidden_dims, out_features=self.hidden_dims // 4)
        self.act3 = nn.ReLU()
        self.dense4 = nn.Linear(in_features=self.hidden_dims // 4, out_features=2)
        self.out_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = x.reshape(x.shape[0], 1, -1)
        x = self.avgpool1d(x)

        self.dense1 = nn.Linear(in_features=x.shape[-1], out_features=self.hidden_dims * 4)

        x = self.dense1(x)
        x = self.act1(x)

        x = self.dense2(x)
        x = self.act2(x)

        x = self.dense3(x)
        x = self.act3(x)

        x = self.dense4(x)
        x = self.out_activation(x)
        return x


class SubMSD(nn.Module):
    def __init__(self, params):
        super(SubMSD, self).__init__()
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.avgpool_params = params["params_avgpool"]
        self.distributor_params = params["params_distributor"]
        self.final_params = params["params_final_msd"]

        self.initiator = MSDInitiator(params=self.avgpool_params)
        self.distributor = Distributor(params=self.distributor_params)
        self.final = MSDFinal(params=self.final_params)

    def forward(self, x):
        x = self.initiator(x)
        initiator_op = x
        x = self.distributor(x)
        distributor_op = x
        x = self.final(x)
        return x, initiator_op, distributor_op


class MSD(nn.Module):
    def __init__(self, params):
        super(MSD, self).__init__()
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kernel_list = self.params["params_distributor"]["kernel_list"]
        self.num_sub_msd = len(self.kernel_list)

    def forward(self, x):
        outs = []
        initiators = []
        distributors = []

        for kernel_size in self.kernel_list:
            self.params["kernel_size"] = kernel_size
            self.submsd_layer = SubMSD(params=self.params)
            out, initiator_op, distributor_op = self.submsd_layer(x)
            outs.append(out)
            initiators.append(initiator_op)
            # print(distributor_op.shape)
            distributors.append(distributor_op)

        x = torch.cat(outs, dim=1)
        initiator_output = torch.cat(initiators, dim=0)
        distributor_output = torch.cat(distributors, dim=1)
        return x, initiator_output, distributor_output


class MCDInitiator(nn.Module):
    def __init__(self, params):
        super(MCDInitiator, self).__init__()
        self.params = params
        self.height = self.params["height"]
        self.width = self.params["width"]

    def forward(self, x):
        x = x.reshape(x.shape[0], self.height, self.width)
        x = torch.unsqueeze(x, dim=1)
        return x


class Convolver(nn.Module):
    def __init__(self, params):
        super(Convolver, self).__init__()
        self.params = params
        self.depth = self.params["depth"]
        self.pool = nn.MaxPool2d(kernel_size=self.params["kernel_size"], stride=2)

    def forward(self, x):
        dpn_resblocks = []
        for i in range(self.depth):
            layer = DPNResBlock(out_channels=self.params["out_channels"], kernel_size=self.params["kernel_size"])
            dpn_resblocks.append(layer)
            self.params["in_channels"] = self.params["out_channels"]
            self.params["out_channels"] = self.params["out_channels"] * 2

        for block in dpn_resblocks:
            x = block(x)
        return x


class MCDFinal(nn.Module):
    def __init__(self, params):
        super(MCDFinal, self).__init__()
        self.params = params
        self.in_features = self.params["in_features"]
        self.hidden_features = self.params["hidden_features"]
        self.out_features = self.params["out_features"]

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=self.in_features, out_features=self.hidden_features)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=self.hidden_features, out_features=self.hidden_features // 2)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=self.hidden_features // 2, out_features=self.hidden_features // 4)
        self.act3 = nn.ReLU()
        self.out = nn.Linear(in_features=self.hidden_features // 4, out_features=self.out_features)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.out_act(self.out(x))
        return x


class SubMCD(nn.Module):
    def __init__(self, params):
        super(SubMCD, self).__init__()
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initiator_params = self.params["initiator_params"]
        self.convolver_params = self.params["convolver_params"]
        self.final_params = self.params["final_params"]

        self.initiator = MCDInitiator(params=self.initiator_params)
        self.convolver = Convolver(params=self.convolver_params)
        self.final = None

    def forward(self, x):
        x = self.initiator(x)
        initiator_op = x
        x = self.convolver(x)
        convolver_op = x

        # Final Config
        self.final_params["in_features"] = x.shape[1] * x.shape[2] * x.shape[3]
        self.final = MCDFinal(params=self.final_params)

        x = self.final(x)
        x = torch.unsqueeze(x, dim=1)
        return x, initiator_op, convolver_op


class MCD(nn.Module):
    def __init__(self, params):
        super(MCD, self).__init__()
        self.params = params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kernel_list = self.params["convolver_params"]["kernel_list"]
        self.num_sub_mcd = len(self.kernel_list)

    def forward(self, x):
        outs = []
        initiators = []
        convolvers = []
        for kernel_size in self.kernel_list:
            self.params["kernel_size"] = kernel_size
            self.submcd_layer = SubMCD(params=self.params)
            out, initiator_op, convolver_op = self.submcd_layer(x)
            outs.append(out)
            initiators.append(initiator_op)
            convolvers.append(convolver_op)
            # print(convolver_op.shape)
        x = torch.cat(outs, dim=1)
        initiator_output = torch.cat(initiators, dim=0)
        convolver_output = torch.cat(convolvers, dim=1)
        return x, initiator_output, convolver_output
