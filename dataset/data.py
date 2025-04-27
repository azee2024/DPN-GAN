import os
import numpy as np
import json
import torch
import librosa
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn


class AudioDataset(Dataset):
    def __init__(self, data_path, data_config, num_points=1000):
        self.data_path = data_path
        self.data_config = data_config
        self.num_points = num_points
        self.file_list = os.listdir(self.data_path)[:num_points]
        self.audio_length = self.data_config["audio_length"]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = os.path.join(self.data_path, self.file_list[idx])
        data = np.load(file_name)

        # Extracting data
        metadata = data["metadata"]
        audio = data["audio"]
        mel_spec = data["mel_spec"]

        # # Calculate the amount of padding needed on each side
        # padding_left = (self.audio_length - len(audio)) // 2
        # padding_right = self.audio_length - len(audio) - padding_left
        #
        # # Pad the array on both sides
        # audio = np.pad(audio, (padding_left, padding_right), mode='constant')
        audio = audio[:self.audio_length]

        # Assuming your data is a numpy array, you can convert it to a PyTorch tensor
        tensor_audio = torch.from_numpy(audio).reshape(-1, 1).to(torch.float32)
        tensor_meta = torch.tensor(metadata).reshape(-1, 1).to(torch.float32)
        tensor_mel_spec = torch.tensor(mel_spec).to(torch.float32)
        return tensor_audio, tensor_meta, tensor_mel_spec

#
# class LJSpeechDataset(Dataset):
#     def __init__(self, data_path):
#         self.data_path = data_path
#         self.file_list = os.listdir(self.data_path)
#
#     def __len__(self):
#         return len(self.file_list)
#
#     def __getitem__(self, idx):
#         file_name = os.path.join(self.data_path, self.file_list[idx])
#         data = np.load(file_name)
#
#         # Extracting data
#         metadata = data["metadata"]
#         audio = data["audio"]
#         mel_spec = data["mel_spec"]
#
#         # Assuming your data is a numpy array, you can convert it to a PyTorch tensor
#         tensor_audio = torch.from_numpy(audio).reshape(-1, 1)
#         tensor_meta = torch.tensor(metadata).reshape(-1, 1)
#         tensor_mel_spec = torch.tensor(mel_spec)
#         return tensor_audio, tensor_meta, tensor_mel_spec
#
#
# class AudioMNISTDataset(Dataset):
#     def __init__(self, data_path):
#         self.data_path = data_path
#         self.file_list = os.listdir(self.data_path)
#
#     def __len__(self):
#         return len(self.file_list)
#
#     def __getitem__(self, idx):
#         file_name = os.path.join(self.data_path, self.file_list[idx])
#         data = np.load(file_name)
#
#         # print(self.file_list[idx])
#         # label, subfolder_index, data_index = self.file_list[idx].split("_")
#
#         # Extracting data
#         metadata = data["metadata"]
#         audio = data["audio"]
#         mel_spec = data["mel_spec"]
#
#         # Assuming your data is a numpy array, you can convert it to a PyTorch tensor
#         tensor_audio = torch.from_numpy(audio).reshape(-1, 1)
#         tensor_meta = torch.tensor(metadata).reshape(-1, 1)
#         tensor_mel_spec = torch.tensor(mel_spec)
#         return tensor_audio, tensor_meta, tensor_mel_spec
#
#
# class LibriSpeechDataset(Dataset):
#     def __init__(self, data_path):
#         self.data_path = data_path
#         self.file_list = os.listdir(self.data_path)
#
#     def __len__(self):
#         return len(self.file_list)
#
#     def __getitem__(self, idx):
#         file_name = os.path.join(self.data_path, self.file_list[idx])
#         data = np.load(file_name)
#
#         # Extracting data
#         metadata = data["metadata"]
#         audio = data["audio"]
#         mel_spec = data["mel_spec"]
#
#         # Assuming your data is a numpy array, you can convert it to a PyTorch tensor
#         tensor_audio = torch.from_numpy(audio).reshape(-1, 1)
#         tensor_meta = torch.tensor(metadata).reshape(-1, 1)
#         tensor_mel_spec = torch.tensor(mel_spec)
#         return tensor_audio, tensor_meta, tensor_mel_spec
#
#
# class VCTKDataset(Dataset):
#     def __init__(self, data_path):
#         self.data_path = data_path
#         self.file_list = os.listdir(self.data_path)
#
#     def __len__(self):
#         return len(self.file_list)
#
#     def __getitem__(self, idx):
#         file_name = os.path.join(self.data_path, self.file_list[idx])
#         data = np.load(file_name)
#
#         # Extracting data
#         metadata = data["metadata"]
#         audio = data["audio"]
#         mel_spec = data["mel_spec"]
#
#         # Assuming your data is a numpy array, you can convert it to a PyTorch tensor
#         tensor_audio = torch.from_numpy(audio).reshape(-1, 1)
#         tensor_meta = torch.tensor(metadata).reshape(-1, 1)
#         tensor_mel_spec = torch.tensor(mel_spec)
#         return tensor_audio, tensor_meta, tensor_mel_spec


def get_dataloader(dataset, batch_size=32, shuffle=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    data_path_audiomnist = "../data/audiomnist/preprocessed_data"
    data_path_ljspeech = "../data/ljspeech/processed_data"
    data_path_librispeech = "../data/librispeech/processed_data"
    data_path_vctk = "../data/vctk/processed_data"

    batch_size = 32
    shuffle = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_config_path_audiomnist = "../usage/data_config_audiomnist.json"
    data_config_path_ljspeech = "../usage/data_config_ljspeech.json"
    data_config_path_librispeech = "../usage/data_config_librispeech.json"
    data_config_path_vctk = "../usage/data_config_vctk.json"

    f_data_config_audiomnist = open(data_config_path_audiomnist, "r")
    data_config_audiomnist = json.load(f_data_config_audiomnist)
    f_data_config_ljspeech = open(data_config_path_ljspeech, "r")
    data_config_ljspeech = json.load(f_data_config_ljspeech)
    f_data_config_librispeech = open(data_config_path_librispeech, "r")
    data_config_librispeech = json.load(f_data_config_librispeech)
    f_data_config_vctk = open(data_config_path_vctk, "r")
    data_config_vctk = json.load(f_data_config_vctk)

    audiomnist_dataset = AudioDataset(data_path=data_path_audiomnist, data_config=data_config_audiomnist)
    audiomnist_dataloader = get_dataloader(audiomnist_dataset, batch_size=batch_size, shuffle=shuffle)

    print("Audiomnist Dataset")
    for i, (real_audio, meta, mel_spec) in enumerate(audiomnist_dataloader):
        # Move data to device
        real_audio = real_audio

        print(real_audio.shape)
        print(mel_spec.shape)
        break

    ljspeech_dataset = AudioDataset(data_path=data_path_ljspeech, data_config=data_config_ljspeech)
    ljspeech_dataloader = get_dataloader(ljspeech_dataset, batch_size=batch_size, shuffle=shuffle)

    print("LJSpeech Dataset:")
    for i, (real_audio, meta, mel_spec) in enumerate(ljspeech_dataloader):
        # Move data to device
        real_audio = real_audio

        print(real_audio.dtype)
        print(mel_spec.dtype)
        break
    pass

    print("LibriSpeech Dataset:")
    librispeech_dataset = AudioDataset(data_path=data_path_librispeech, data_config=data_config_librispeech)
    librispeech_dataloader = get_dataloader(librispeech_dataset, batch_size=batch_size, shuffle=shuffle)

    for i, (real_audio, meta, mel_spec) in enumerate(librispeech_dataloader):
        # Move data to device
        real_audio = real_audio

        print(real_audio.dtype)
        print(mel_spec.dtype)
        break
    pass

    print("VCTK Dataset:")
    vctk_dataset = AudioDataset(data_path=data_path_vctk, data_config=data_config_vctk)
    vctk_dataloader = get_dataloader(vctk_dataset, batch_size=batch_size, shuffle=shuffle)

    for i, (real_audio, meta, mel_spec) in enumerate(vctk_dataloader):
        # Move data to device
        real_audio = real_audio

        print(real_audio.shape)
        print(mel_spec.shape)
        break
    pass
