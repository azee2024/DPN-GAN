import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa
MAX_LENGTH = 90001


def load(file_):
    y, sampling_rate = librosa.load(file_, sr=3000, offset=0.0, duration=30)

    if len(y) < MAX_LENGTH:
        # Calculate the amount of padding needed on each side
        padding_left = (MAX_LENGTH - len(y)) // 2
        padding_right = MAX_LENGTH - len(y) - padding_left

        # Pad the array on both sides
        y = np.pad(y, (padding_left, padding_right), mode='constant')
    y = y.reshape(1, -1)
    return y


def process_gtzan_dataset(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metadata_path = os.path.join(input_dir, "features_30_sec.csv")

    metadata = pd.read_csv(metadata_path)
    _ = metadata.pop("label")

    mel_dir = os.path.join(input_dir, "images_original")

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".wav"):
                # Load audio file
                # Image path
                mel_filename = "".join(filename.split(".")[:2]) + "." + "png"
                mel_label = filename.split(".")[0]
                mel_path = os.path.join(mel_dir, mel_label, mel_filename)

                # Create output directory if it doesn't exist
                if os.path.exists(mel_path):
                    mel_spec = plt.imread(mel_path)

                    audio_path = os.path.join(root, filename)

                    y = load(audio_path)

                    # Filter the DataFrame based on the specified column value
                    filtered_meta = metadata[metadata["filename"] == filename].values
                    filtered_meta = np.array(filtered_meta[1:])
                    filtered_meta = filtered_meta.reshape(1, -1)

                    np.savez(os.path.join(output_dir, ".".join(filename.split(".")[:2])) + ".npz",
                             mel_spec=mel_spec,
                             audio=y,
                             metadata=filtered_meta)


if __name__ == "__main__":
    # Define input and output directories
    input_dir = "../data/gtzan/Data"
    output_dir = "../data/gtzan/processed_data"
    # Process the dataset
    process_gtzan_dataset(input_dir, output_dir)
