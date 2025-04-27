import torch
import torch.nn as nn

# Generator Imports
from models.generator_utils import MelInitiatorBlock, MetaInitiatorBlock, MelMetaConcatenator, Modulator, ProcessorBlock

# Discriminator Imports
from models.discriminator_utils import MSD, MCD


class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()

        self.params = params
        self.device = "cpu"  # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mel_initiator_params = params["mel_initiator_params"]
        self.meta_initiator_params = params["meta_initiator_params"]
        self.modulator_params = None
        self.processor_params = None
        self.audio_length = self.mel_initiator_params["audio_length"]

        self.mel_initiator_block = MelInitiatorBlock(params=self.mel_initiator_params)
        self.meta_initiator_block = MetaInitiatorBlock(params=self.meta_initiator_params)
        self.mel_meta_concatenator = MelMetaConcatenator()
        self.modulator_block = None
        self.processor_block = None

    def forward(self, x_mel, x_meta):
        x_mel = self.mel_initiator_block(x_mel)
        x_meta = self.meta_initiator_block(x_meta)
        x = self.mel_meta_concatenator(x_mel, x_meta)

        # Defining the Modulator
        convt_in_channels = x.shape[1]
        convt_out_channels = convt_in_channels
        convt_kernel_size = 3
        in_channels = convt_out_channels
        out_channels = in_channels * 2
        kernel_size = 3
        lstm_num_layers = 2
        shape = x.shape

        self.modulator_params = {
            "ConvT_in_channels": convt_in_channels,
            "ConvT_out_channels": convt_out_channels,
            "ConvT_kernel_size": convt_kernel_size,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "lstm_num_layers": lstm_num_layers,
            "shape": shape,
            "dpn_depth": self.mel_initiator_params["dpn_depth"]
        }


        self.modulator_block = Modulator(params=self.modulator_params)

        # Forward pass
        x = self.modulator_block(x)

        # Defining the Processor Block
        shape = x.shape

        self.processor_params = {
            "in_channels": shape[1],
            "out_channels": shape[1] * 2,
            "kernel_size": kernel_size,
            "shape": shape,
            "out_seq_length": self.audio_length
        }

        self.processor_block = ProcessorBlock(params=self.processor_params)
        x = self.processor_block(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.msd_params = params["msd_params"]
        self.mcd_params = params["mcd_params"]

        self.msd = MSD(params=self.msd_params)
        self.mcd = MCD(params=self.mcd_params)

    def forward(self, x):
        x_msd, x_initiators_msd, x_distributors_msd = self.msd(x)
        x_mcd, x_initiators_mcd, x_convolvers_mcd = self.mcd(x)
        return x_msd, x_mcd, [x_initiators_msd, x_distributors_msd], [x_initiators_mcd, x_convolvers_mcd]


if __name__ == "__main__":
    # Generator
    # Shape of the Mel Spectrogram
    batch_size = 16
    num_channels = 1
    mel_filters = 128
    timeframes = 100
    audio_length = 47998
    shape = (batch_size, num_channels, mel_filters, timeframes)

    in_channels = 1
    out_channels = 32
    kernel_size = 3

    lstm_hidden_size = 64
    lstm_num_layers = 2

    # Create random input tensor with the shape (batch_size, num_channels, mel_filters, timeframes)
    x_mel = torch.randn(batch_size, num_channels, mel_filters, timeframes)
    print("Mel Initiator Input Shape: ", x_mel.shape)

    # Instantiate the model
    params_mel_initiator_block = {
        "in_channels": 1,
        "out_channels": 32,
        "kernel_size": 3,
        "lstm_hidden_size": 64,
        "lstm_num_layers": 2,
        "input_shape": shape
    }

    # Define model and input parameters
    batch_size = 16
    input_size = 10  # Input size
    hidden_size = 64  # Hidden layer size
    output_size = 32  # Output size

    # Generate random input
    x_meta = torch.randn(batch_size, input_size)
    print("Meta Initiator Input Shape: ", x_meta.shape)

    # Create model
    params_metainitiator_block = {
        "input_size": 10,
        "hidden_size": 64,
        "output_size": 32
    }

    params_generator = {
        "mel_initiator_params": params_mel_initiator_block,
        "meta_initiator_params": params_metainitiator_block
    }

    generator = Generator(params=params_generator)

    out_gen = generator(x_mel, x_meta)
    print("Generator Output Shape: ", out_gen.shape)

    # Discriminator

    # Discriminator
    batch_size = 16

    # Inputs
    generated_samples = torch.randn(batch_size, 1, audio_length)
    real_samples = torch.randn(batch_size, 1, audio_length)

    # Labels
    labels_generated = torch.zeros(batch_size, dtype=torch.long)  # Label 0
    labels_real = torch.ones(batch_size, dtype=torch.long)  # Label 1

    # MSD Block
    # Initiator
    params_avgpool_discriminator = {
        "kernel_size": 11,
        "stride": 4
    }

    # Distributor
    params_distributor_msd_discrimiantor = {
        "defconv_in_channels": 1,
        "defconv_out_channels": 16,
        "kernel_size": 3,
        "kernel_size_list": [3, 5, 7, 9, 11],
        "stride": 1,
        "depth": 4
    }

    # Final MSD Layers
    params_final_msd = {
        "kernel_size": 7,
        "stride": 4,
        "hidden_dims": 512
    }

    params_msd = {
        "params_avgpool": params_avgpool_discriminator,
        "params_distributor": params_distributor_msd_discrimiantor,
        "params_final_msd": params_final_msd
    }

    # MCD
    mcd_initiator_params = {
        "height": 103,
        "width": 466
    }

    convolver_params = {
        "in_channels": 1,
        "out_channels": 8,
        "kernel_size": 3,
        "kernel_size_list": [3, 5, 7, 9, 11],
        "stride": 1,
        "depth": 3
    }

    mcd_final_params = {
        "in_features": 0,  # To be modified later
        "hidden_features": 512,
        "out_features": 2
    }

    mcd_params = {
        "initiator_params": mcd_initiator_params,
        "convolver_params": convolver_params,
        "final_params": mcd_final_params
    }

    discriminator_params = {
        "msd_params": params_msd,
        "mcd_params": mcd_params
    }

    discriminator_model = Discriminator(params=discriminator_params)
    discriminator_output_msd, discriminator_output_mcd = discriminator_model(generated_samples)
    print("Discriminator MSD Output Shape: ", discriminator_output_msd.shape)
    print("Discriminator MCD Output Shape: ", discriminator_output_mcd.shape)
