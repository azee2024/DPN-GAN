import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchaudio


class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.criterion = F.binary_cross_entropy_with_logits

    def forward(self, y_pred, y_true):
        loss = self.criterion(y_pred, y_true)
        return loss


class MelSpectrogramLoss(nn.Module):
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256, n_mels=128):
        super(MelSpectrogramLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                                  n_fft=n_fft,
                                                                  hop_length=hop_length,
                                                                  n_mels=n_mels)

    def forward(self, real_audio_seq, fake_audio_seq):
        real_mel_spec_transform = self.mel_transform(real_audio_seq)
        fake_mel_spec_transform = self.mel_transform(fake_audio_seq)

        loss = F.mse_loss(real_mel_spec_transform, fake_mel_spec_transform)
        return loss


class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super(FeatureMatchingLoss, self).__init__()

    def forward(self,
                discriminator_output_msd_initiator_features_generated,
                discriminator_output_msd_initiator_features_real,
                discriminator_output_msd_distributor_features_generated,
                discriminator_output_msd_distributor_features_real,
                discriminator_output_mcd_initiator_features_generated,
                discriminator_output_mcd_initiator_features_real,
                discriminator_output_mcd_convolver_features_generated,
                discriminator_output_mcd_convolver_features_real):
        loss_discriminator_output_msd_initiator_features = F.mse_loss(discriminator_output_msd_initiator_features_generated, discriminator_output_msd_initiator_features_real)
        # print(discriminator_output_msd_distributor_features_generated.shape, discriminator_output_msd_distributor_features_real.shape)
        loss_discriminator_output_msd_distributor_features = F.mse_loss(discriminator_output_msd_distributor_features_generated, discriminator_output_msd_distributor_features_real)
        loss_discriminator_output_mcd_initiator_features = F.mse_loss(discriminator_output_mcd_initiator_features_generated, discriminator_output_mcd_initiator_features_real)
        loss_discriminator_output_mcd_convolver_features = F.mse_loss(discriminator_output_mcd_convolver_features_generated, discriminator_output_mcd_convolver_features_real)

        loss_discriminator_msd_feature = loss_discriminator_output_msd_initiator_features + loss_discriminator_output_msd_distributor_features
        loss_discriminator_mcd_feature = loss_discriminator_output_mcd_initiator_features + loss_discriminator_output_mcd_convolver_features
        feature_loss = loss_discriminator_msd_feature + loss_discriminator_mcd_feature
        return feature_loss


def create_discriminator_labels_gen(disc_op):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Creatng Labels
    disc_op_label_ones = torch.ones(size=[disc_op.shape[0],
                                          disc_op.shape[1], 1],
                                    dtype=torch.long)
    # Convert to one-hot encoding
    disc_op_label = torch.zeros(disc_op.shape[0],
                                disc_op.shape[1],
                                2)
    # Scatter the ones to the appropriate locations
    disc_op_label.scatter_(2, disc_op_label_ones, 1)
    return disc_op_label


class GeneratorLoss(nn.Module):
    def __init__(self, discriminator):
        super(GeneratorLoss, self).__init__()
        # self.discriminator = discriminator
        self.adversarial_loss = AdversarialLoss()
        self.mel_loss = MelSpectrogramLoss()
        self.feature_matching_loss = FeatureMatchingLoss()
        self.lambda_fm = 0.2
        self.lambda_mel = 0.7

    def forward(self,
                real_audio,
                fake_audio,
                discriminator_output_msd_generated,
                discriminator_output_mcd_generated,
                discriminator_output_msd_initiator_features_generated,
                discriminator_output_msd_distributor_features_generated,
                discriminator_output_mcd_initiator_features_generated,
                discriminator_output_mcd_convolver_features_generated,
                discriminator_output_msd_initiator_features_real,
                discriminator_output_msd_distributor_features_real,
                discriminator_output_mcd_initiator_features_real,
                discriminator_output_mcd_convolver_features_real):

        # Creatng MCD Labels
        discriminator_output_msd_generated_label = create_discriminator_labels_gen(discriminator_output_msd_generated)
        discriminator_output_mcd_generated_label = create_discriminator_labels_gen(discriminator_output_mcd_generated)

        adv_loss_disc_msd_generated = self.adversarial_loss(discriminator_output_msd_generated,
                                                            discriminator_output_msd_generated_label)
        adv_loss_disc_mcd_generated = self.adversarial_loss(discriminator_output_mcd_generated,
                                                            discriminator_output_mcd_generated_label)
        adv_loss = adv_loss_disc_msd_generated + adv_loss_disc_mcd_generated
        mel_loss = self.mel_loss(fake_audio, real_audio)
        fm_loss = self.feature_matching_loss(discriminator_output_msd_initiator_features_generated,
                                             discriminator_output_msd_initiator_features_real,
                                             discriminator_output_msd_distributor_features_generated,
                                             discriminator_output_msd_distributor_features_real,
                                             discriminator_output_mcd_initiator_features_generated,
                                             discriminator_output_mcd_initiator_features_real,
                                             discriminator_output_mcd_convolver_features_generated,
                                             discriminator_output_mcd_convolver_features_real)
        gen_loss = adv_loss + self.lambda_fm * fm_loss + self.lambda_mel * mel_loss
        return gen_loss, adv_loss, mel_loss, fm_loss


def create_discriminator_label_disc(disc_op, type):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if type == "generated":
        disc_op_label_zeros = torch.zeros(size=[disc_op.shape[0],
                                                disc_op.shape[1],
                                                1],
                                          dtype=torch.long)
        disc_op_label = torch.ones(disc_op.shape[0],
                                   disc_op.shape[1],
                                   2)
        disc_op_label.scatter_(2, disc_op_label_zeros, 1)

    elif type == "real":
        # Creatng Labels
        disc_op_label_ones = torch.ones(size=[disc_op.shape[0],
                                              disc_op.shape[1],
                                              1],
                                        dtype=torch.long)
        # Convert to one-hot encoding
        disc_op_label = torch.zeros(disc_op.shape[0],
                                    disc_op.shape[1],
                                    2)

        # Scatter the ones to the appropriate locations
        disc_op_label.scatter_(2, disc_op_label_ones, 1)
    else:
        return -1
    return disc_op_label


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.adversarial_loss = AdversarialLoss()

    def forward(self,
                discriminator_output_msd_generated,
                discriminator_output_msd_real,
                discriminator_output_mcd_generated,
                discriminator_output_mcd_real):
        discriminator_output_msd_generated_label = create_discriminator_label_disc(discriminator_output_msd_generated,
                                                                                   type="generated")
        discriminator_output_msd_real_label = create_discriminator_label_disc(discriminator_output_msd_real,
                                                                              type="real")
        discriminator_output_mcd_generated_label = create_discriminator_label_disc(discriminator_output_mcd_generated,
                                                                                   type="generated")
        discriminator_output_mcd_real_label = create_discriminator_label_disc(discriminator_output_mcd_real,
                                                                              type="real")

        msd_loss_generated = self.adversarial_loss(discriminator_output_msd_generated,
                                                   discriminator_output_msd_generated_label)
        msd_loss_real = self.adversarial_loss(discriminator_output_msd_real,
                                              discriminator_output_msd_real_label)
        msd_loss = msd_loss_real + msd_loss_generated

        mcd_loss_generated = self.adversarial_loss(discriminator_output_mcd_generated,
                                                   discriminator_output_mcd_generated_label)
        mcd_loss_real = self.adversarial_loss(discriminator_output_mcd_real,
                                              discriminator_output_mcd_real_label)
        mcd_loss = mcd_loss_real + mcd_loss_generated

        disc_loss = msd_loss + mcd_loss

        return disc_loss, msd_loss, mcd_loss


# Mock discriminator class for feature extraction
class MockDiscriminator(nn.Module):
    def __init__(self):
        super(MockDiscriminator, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)

    def extract_features(self, x):
        return [x]  # Dummy feature extraction


if __name__ == "__main__":
    batch_size = 16
    audio_seq_length = 47998
    generator_output_shape = [batch_size, 1, audio_seq_length]

    mel_channels = 1
    mel_filters = 128
    mel_timeframes = 100
    mel_spectrogram_shape = [batch_size, mel_channels, mel_filters, mel_timeframes]

    msd_disc_shape = [batch_size, 1, 2]
    mcd_disc_shape = [batch_size, 1, 2]

    x_mel = torch.randn(mel_spectrogram_shape)
    x_generated = torch.randn(generator_output_shape)

    labels = torch.sigmoid(torch.randn(msd_disc_shape))
    msd_disc_output = torch.sigmoid(torch.randn(msd_disc_shape))
    mcd_disc_output = torch.sigmoid(torch.randn(mcd_disc_shape))
