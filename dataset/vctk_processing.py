import os
import numpy as np
import librosa
from sentence_transformers import SentenceTransformer


def process_metadata(data_dir):
    texts = []
    filenames = []
    meta_dir = os.path.join(data_dir, "txt")
    output_embedding_dir = os.path.join("/".join(data_dir.split("/")[:-1]), "text_embeddings")
    for root, dirs, files in os.walk(meta_dir):
        for filename in files:
            if filename.endswith(".txt"):
                filepath = os.path.join(root, filename)
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

    # Initialize the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create embeddings
    embeddings = model.encode(texts)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_embedding_dir):
        os.makedirs(output_embedding_dir)

    # Save embeddings in .npz file
    for filename, embedding in zip(filenames, embeddings):
        npz_path = os.path.join(output_embedding_dir, filename.split(".")[0] + ".npz")
        np.savez(npz_path, filename=filename, embedding=embedding)
    pass


def process_data(data_dir):
    wav_dir = os.path.join(data_dir, "wav48")
    meta_dir = os.path.join("/".join(data_dir.split("/")[:-1]), "text_embeddings")
    output_dir = os.path.join("/".join(data_dir.split("/")[:-1]), "processed_data")
    MAX_LENGTH = 925597

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(wav_dir):
        for filename in files:
            if filename.endswith(".wav"):
                audio_file_name = filename
                meta_file_name = filename.split(".")[0] + ".npz"

                # Load audio file
                audio_path = os.path.join(root, audio_file_name)
                y, sr = librosa.load(audio_path, sr=None)
                if y.shape[0] > MAX_LENGTH:
                    MAX_LENGTH = y.shape[0]

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


if __name__ == "__main__":
    data_dir = "../data/vctk/VCTK-Corpus/VCTK-Corpus"
    process_data(data_dir)

