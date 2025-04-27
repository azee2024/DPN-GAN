import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.models import Generator, Discriminator
from models.losses import GeneratorLoss, DiscriminatorLoss


def do_train(dataloader,
             run_config,
             data_config,
             model_config,
             output_config,
             do_save=False):
    epochs = run_config["epochs"]
    batch_size = data_config["batch_size"]
    audio_length = data_config["audio_length"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    rand_tensor = torch.randn([batch_size, 1, audio_length])
    params_generator = model_config["generator_params"]
    params_discriminator = model_config["discriminator_params"]

    generator = Generator(params=params_generator)

    discriminator = Discriminator(params=params_discriminator)
    discriminator(rand_tensor)

    generator_loss = GeneratorLoss(discriminator=discriminator)
    discriminator_loss = DiscriminatorLoss()

    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr=run_config["generator_learning_rate"],
                                   betas=(0.5, 0.999))

    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                   lr=run_config["discriminator_learning_rate"],
                                   betas=(0.5, 0.999))

    generator_loss_epoch = []
    adv_loss_epoch = []
    mel_loss_epoch = []
    fm_loss_epoch = []

    discriminator_loss_epoch = []
    msd_loss_epoch = []
    mcd_loss_epoch = []

    for epoch in range(epochs):
        generator_loss_batch = 0
        adv_loss_batch = 0
        mel_loss_batch = 0
        fm_loss_batch = 0

        discriminator_loss_batch = 0
        msd_loss_batch = 0
        mcd_loss_batch = 0

        for (real_audio,
             meta,
             mel_spec) in tqdm(dataloader):
            # Move data to device
            real_audio = real_audio
            real_audio = real_audio.transpose(1, 2)

            real_audio = real_audio

            x_mel = torch.unsqueeze(mel_spec, dim=1)
            x_meta = meta.transpose(1, 2).to(torch.float32)

            generated_audio = generator(x_mel, x_meta)
            # generated_audio_numpy = generated_audio.cpu().detach().numpy()

            # print("Generated Audio Shape: ", generated_audio.shape)

            # Concatenate generated and real audio
            combined_audio = torch.cat((generated_audio.detach(), real_audio.detach()), dim=0)

            # Pass through the discriminator
            combined_discriminator_output_msd, \
            combined_discriminator_output_mcd, \
            combined_discriminator_output_msd_features, \
            combined_discriminator_output_mcd_features = discriminator(combined_audio)

            # Split the outputs
            batch_size = real_audio.size(0)
            discriminator_output_msd_generated = combined_discriminator_output_msd[:batch_size]
            discriminator_output_msd_real = combined_discriminator_output_msd[batch_size:]

            discriminator_output_mcd_generated = combined_discriminator_output_mcd[:batch_size]
            discriminator_output_mcd_real = combined_discriminator_output_mcd[batch_size:]

            discriminator_output_msd_initiator_features_combined = combined_discriminator_output_msd_features[0]
            discriminator_output_msd_distributor_features_combined = combined_discriminator_output_msd_features[1]
            discriminator_output_mcd_initiator_features_combined = combined_discriminator_output_mcd_features[0]
            discriminator_output_mcd_convolver_features_combined = combined_discriminator_output_mcd_features[1]

            discriminator_output_msd_initiator_features_generated = discriminator_output_msd_initiator_features_combined[
                                                                    :batch_size]
            discriminator_output_msd_initiator_features_real = discriminator_output_msd_initiator_features_combined[
                                                               batch_size:]

            discriminator_output_msd_distributor_features_generated = discriminator_output_msd_distributor_features_combined[
                                                                      :batch_size]
            discriminator_output_msd_distributor_features_real = discriminator_output_msd_distributor_features_combined[
                                                                 batch_size:]

            discriminator_output_mcd_initiator_features_generated = discriminator_output_mcd_initiator_features_combined[
                                                                    :batch_size]
            discriminator_output_mcd_initiator_features_real = discriminator_output_mcd_initiator_features_combined[
                                                               batch_size:]

            discriminator_output_mcd_convolver_features_generated = discriminator_output_mcd_convolver_features_combined[
                                                                    :batch_size]
            discriminator_output_mcd_convolver_features_real = discriminator_output_mcd_convolver_features_combined[
                                                               batch_size:]

            # Train the Generator
            optimizer_G.zero_grad()
            # Calculate the generator loss
            gen_loss, adv_loss, mel_loss, fm_loss = \
                generator_loss(real_audio,
                               generated_audio,
                               discriminator_output_msd_generated,
                               discriminator_output_mcd_generated,
                               discriminator_output_msd_initiator_features_generated,
                               discriminator_output_msd_distributor_features_generated,
                               discriminator_output_mcd_initiator_features_generated,
                               discriminator_output_mcd_convolver_features_generated,
                               discriminator_output_msd_initiator_features_real,
                               discriminator_output_msd_distributor_features_real,
                               discriminator_output_mcd_initiator_features_real,
                               discriminator_output_mcd_convolver_features_real)
            generator_loss_batch += gen_loss.item()
            adv_loss_batch += adv_loss.item()
            mel_loss_batch += mel_loss.item()
            fm_loss_batch += fm_loss.item()

            # BackProp
            gen_loss.backward(retain_graph=True)
            # Update Optimizer
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()

            # Calculate the discriminator loss
            disc_loss, msd_loss, mcd_loss = discriminator_loss(discriminator_output_msd_generated,
                                                               discriminator_output_msd_real,
                                                               discriminator_output_mcd_generated,
                                                               discriminator_output_mcd_real)
            discriminator_loss_batch += disc_loss.item()
            msd_loss_batch += msd_loss.item()
            mcd_loss_batch += mcd_loss.item()

            disc_loss.backward()
            optimizer_D.step()
            # break

        generator_loss_epoch.append(generator_loss_batch)
        adv_loss_epoch.append(adv_loss_batch)
        mel_loss_epoch.append(mel_loss_batch)
        fm_loss_epoch.append(fm_loss_batch)

        discriminator_loss_epoch.append(discriminator_loss_batch)
        msd_loss_epoch.append(msd_loss_batch)
        mcd_loss_epoch.append(mcd_loss_batch)

        if epoch % 10 == 0:
            print()
            print("Epoch: {} ; Generator Loss: {} ; Discriminator Loss: {}".format(epoch,
                                                                                   generator_loss_batch,
                                                                                   discriminator_loss_batch))
        # break

    if do_save:
        i = 0
        real_audios = []
        generated_audios = []

        for (real_audio,
             meta,
             mel_spec) in tqdm(dataloader):
            # Move data to device
            real_audio = real_audio
            real_audio = real_audio.transpose(1, 2)

            x_mel = torch.unsqueeze(mel_spec, dim=1)
            x_meta = meta.transpose(1, 2).to(torch.float32)

            generated_audio = generator(x_mel, x_meta)

            real_audios.append(real_audio.detach().numpy())
            generated_audios.append(generated_audio.detach().numpy())
            i += 1

        epoch_losses_save_path = os.path.join(output_config["output_dir"], "epoch_losses.npz")
        np.savez(file=epoch_losses_save_path,
                 generator_losses=np.array(generator_loss_epoch),
                 adv_losses=np.array(adv_loss_epoch),
                 mel_losses=np.array(mel_loss_epoch),
                 fm_losses=np.array(fm_loss_epoch),
                 discriminator_loss=np.array(discriminator_loss_epoch),
                 msd_losses=np.array(msd_loss_epoch),
                 mcd_losses=np.array(mcd_loss_epoch))

        audio_save_path = os.path.join(output_config["output_dir"], "audio_save_path.npz")
        np.savez(file=audio_save_path,
                 real_audio=real_audios,
                 generated_audio=generated_audios)


if __name__ == "__main__":
    # Data Preparation
    data_params = {
        "batch_size": 2,
        "num_channels": 1,
        "mel_filters": 128,
        "timeframes": 100,
        "audio_length": 47998
    }

    batch_size = data_params["batch_size"]
    num_channels = data_params["num_channels"]
    mel_filters = data_params["mel_filters"]
    timeframes = data_params["timeframes"]
    audio_length = data_params["audio_length"]
    shape = (batch_size, num_channels, mel_filters, timeframes)

    # Create random input tensor with the shape (batch_size, num_channels, mel_filters, timeframes)
    x_mel = torch.randn(batch_size, num_channels, mel_filters, timeframes)
    print("Mel Initiator Input Shape: ", x_mel.shape)

    params = {
        "generator_params": {
            "mel_initiator_params": {
                "in_channels": 1,
                "out_channels": 32,
                "kernel_size": 3,
                "lstm_hidden_size": 64,
                "lstm_num_layers": 2,
                "dpn_depth": 3,
                "audio_length": 47998,
                "input_shape": shape
            },
            "meta_initiator_params": {
                "input_size": 10,
                "hidden_size": 64,
                "output_size": 32
            },
        },
        "discriminator_params": {
            "msd_params": {
                "params_avgpool": {
                    "kernel_size": 11,
                    "stride": 4
                },
                "params_distributor": {
                    "defconv_in_channels": 1,
                    "defconv_out_channels": 16,
                    "kernel_size": 3,
                    "kernel_list": [3, 5],
                    "stride": 1,
                    "depth": 4
                },
                "params_final_msd": {
                    "kernel_size": 7,
                    "stride": 4,
                    "hidden_dims": 512
                }
            },
            "mcd_params": {
                "initiator_params": {
                    "height": 103,
                    "width": 466
                },
                "convolver_params": {
                    "in_channels": 1,
                    "out_channels": 8,
                    "kernel_size": 3,
                    "kernel_list": [3, 5],
                    "stride": 1,
                    "depth": 3
                },
                "final_params": {
                    "in_features": 0,  # To be modified later
                    "hidden_features": 512,
                    "out_features": 2
                }
            }
        }

    }

    # Generator Params
    params_generator = params["generator_params"]

    # Discriminator Params
    discriminator_params = params["discriminator_params"]

    # Define model and input parameters
    input_size = 10  # Input size

    # Generate random input
    x_meta = torch.randn(batch_size, input_size)
    print("Meta Initiator Input Shape: ", x_meta.shape)

    generator = Generator(params=params_generator)

    out_gen = generator(x_mel, x_meta)
    print("Generator Output Shape: ", out_gen.shape)

    # Inputs
    generated_samples = torch.randn(batch_size, 1, audio_length)
    real_samples = torch.randn(batch_size, 1, audio_length)

    # Labels
    # labels_generated = torch.zeros(batch_size, dtype=torch.long)  # Label 0
    # labels_real = torch.ones(batch_size, dtype=torch.long)  # Label 1

    discriminator_model = Discriminator(params=discriminator_params)

    discriminator_output_msd_generated, discriminator_output_mcd_generated, \
    discriminator_params_msd_generated, discriminator_params_mcd_generated = discriminator_model(generated_samples)
    discriminator_output_msd_real, discriminator_output_mcd_real = discriminator_output_msd_generated, discriminator_output_mcd_generated  # discriminator_model(generated_samples)
    print("Discriminator MSD Output Shape: ", discriminator_output_msd_generated.shape)
    print("Discriminator MCD Output Shape: ", discriminator_output_mcd_generated.shape)
#
#     # generator_loss = GeneratorLoss(discriminator=discriminator_model)
#     # discriminator_loss = DiscriminatorLoss()
#     #
#     # # Calculate the generator loss
#     # gen_loss = generator_loss(real_samples, generated_samples, x_mel, discriminator_output_msd_generated)
#     # print(f"Generator Loss: {gen_loss.item()}")
#     #
#     # # Calculate the discriminator loss
#     # disc_loss = discriminator_loss(discriminator_output_msd_real, discriminator_output_msd_generated)
#     # print(f"Discriminator Loss: {disc_loss.item()}")
