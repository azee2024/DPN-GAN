import os
import numpy as np
import pandas as pd
import librosa
from sentence_transformers import SentenceTransformer


def read_texts_from_directory(directory):
    texts = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    texts.append(file.read())
            except UnicodeDecodeError:
                try:
                    with open(filepath, 'r', encoding='latin-1') as file:
                        texts.append(file.read())
                except UnicodeDecodeError:
                    with open(filepath, 'r', errors='ignore') as file:
                        texts.append(file.read())
            filenames.append(filename)
    return texts, filenames


def create_and_save_embeddings(input_dir, output_dir, model):
    # Read texts
    texts, filenames = read_texts_from_directory(input_dir)

    # Create embeddings
    embeddings = model.encode(texts)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save embeddings in .npz file
    for filename, embedding in zip(filenames, embeddings):
        npz_path = os.path.join(output_dir, filename.split(".")[0] + ".npz")
        np.savez(npz_path, filename=filename, embedding=embedding)


def process_metadata(meta_path):
    # First Creating the text directory
    output_text_dir = os.path.join(os.path.dirname(meta_path), "output_texts")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_text_dir):
        os.makedirs(output_text_dir)

    data = pd.read_csv(meta_path, sep='|', header=None, dtype=str)

    # Iterate over each row to create a text file for each entry
    for index, row in data.iterrows():
        filepath = os.path.join(output_text_dir, row[0] + '.txt')  # Extract the first 10 characters for the filename
        content = str(row[1])  # Combine the text columns
        with open(filepath, 'w') as file:
            file.write(content)

    output_embedding_dir = os.path.join(os.path.dirname(meta_path), "text_embeddings")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_embedding_dir):
        os.makedirs(output_embedding_dir)

    # Initialize the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create and save embeddings
    create_and_save_embeddings(output_text_dir, output_embedding_dir, model)

    print("Embedding Saved!")


def process_data(data_dir, metadata_path):
    wav_dir = os.path.join(data_dir, "wavs")
    meta_dir = os.path.join(data_dir, "text_embeddings")
    output_dir = os.path.join(data_dir, "processed_data")
    MAX_LENGTH = 222621

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Processing Metadata
    process_metadata(metadata_path)

    for root, dirs, files in os.walk(wav_dir):
        for filename in files:
            if filename.endswith(".wav"):
                audio_file_name = filename
                meta_file_name = filename.split(".")[0] + ".npz"

                # Load audio file
                audio_path = os.path.join(root, audio_file_name)
                y, sr = librosa.load(audio_path, sr=None)

                # Calculate the amount of padding needed on each side
                padding_left = (MAX_LENGTH - len(y)) // 2
                padding_right = MAX_LENGTH - len(y) - padding_left

                # Pad the array on both sides
                y = np.pad(y, (padding_left, padding_right), mode='constant')

                meta = np.load(os.path.join(meta_dir, meta_file_name))
                embedding = meta["embedding"]

                # Perform feature extraction (e.g., Mel spectrogram)
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)

                filename2 = filename.split(".")[0]
                # Save audio data as numpy array
                np.savez(file=os.path.join(output_dir, f"{filename2}.npz"),
                         metadata=embedding,
                         audio=y,
                         mel_spec=mel_spec)
    print("Data Processed!!")


if __name__ == "__main__":
    metadata_path = "../data/ljspeech/LJSpeech-1.1/metadata.csv"
    data_dir = "../data/ljspeech/LJSpeech-1.1"
    process_data(data_dir=data_dir, metadata_path=metadata_path)

    # data_file = "../LJSpeech-1.1/LJSpeech-1.1/processed_data/LJ001-0002.npz"
    # data = np.load(data_file)
    # print("Audio Shape: ", data["audio"].shape)
    # print("Metadata Shape: ", data["metadata"].shape)
    # print("Mel Spectrogram Shape: ", data["mel_spec"].shape)
