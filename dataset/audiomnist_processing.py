import os
import json
import librosa
import numpy as np
import pandas as pd


# Define function to load audio files and save as numpy arrays
def process_dataset(input_dir, output_dir, metadata_path):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    MAX_LENGTH = 47998

    metadata = pd.read_csv(metadata_path, index_col=0)
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".wav"):
                # Load audio file
                # print("Present")
                audio_path = os.path.join(root, filename)
                label, subfolder_index, data_index = filename.split("_")
                label = int(label)
                subfolder_index = int(subfolder_index)
                data_index = int(data_index.split(".")[0])
                y, sr = librosa.load(audio_path, sr=None)

                # Calculate the amount of padding needed on each side
                padding_left = (MAX_LENGTH - len(y)) // 2
                padding_right = MAX_LENGTH - len(y) - padding_left

                # Pad the array on both sides
                y = np.pad(y, (padding_left, padding_right), mode='constant')

                # Filter the DataFrame based on the specified column value
                filtered_meta = metadata[metadata["index"] == int(subfolder_index)].values

                # Perform feature extraction (e.g., Mel spectrogram)
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

                # Save audio data as numpy array
                np.savez(file=os.path.join(output_dir, f"{label}_{subfolder_index}_{data_index}.npz"),
                         metadata=filtered_meta,
                         audio=y,
                         mel_spec=mel_spec,
                         mel_spec_db=mel_spec_db)


if __name__ == "__main__":
    # Define input and output directories
    input_dir = "../data/audiomnist"
    output_dir = "../data/audiomnist/preprocessed_data"
    # Process the dataset
    process_dataset(input_dir, output_dir, metadata_path="../data/audiomnist/metadata.csv")
