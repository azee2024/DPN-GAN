import os
import numpy as np
import librosa
import soundfile as sf
from sentence_transformers import SentenceTransformer
MAX_LENGTH = 392400


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


def save_texts_by_identifier(input_file, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the input file
    with open(input_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # Split the line into identifier and text
        parts = line.strip().split(' ', 1)
        if len(parts) < 2:
            continue  # Skip lines that don't have both identifier and text

        identifier, text = parts
        # Create a filename based on the identifier
        filename = os.path.join(output_dir, f"{identifier}.txt")

        # Save the text to a file
        with open(filename, 'w') as text_file:
            text_file.write(text)


def process_data(data_dir):
    # audio_dir = os.path.join("/".join(data_dir.split("/")[:-1]), "audio_dir")
    output_dir = os.path.join("/".join(data_dir.split("/")[:-1]), "processed_data")
    output_embedding_dir = os.path.join("/".join(data_dir.split("/")[:-1]), "text_embeddings")
    MAX_LENGTH = 522320

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith(".flac"):
                audio_file_name = filename
                meta_file_name = filename.split(".")[0] + ".npz"

                # Load audio file
                audio_path = os.path.join(root, audio_file_name)
                y, sr = sf.read(audio_path)
                if y.shape[0] > MAX_LENGTH:
                    MAX_LENGTH = y.shape[0]

                # Calculate the amount of padding needed on each side
                padding_left = (MAX_LENGTH - len(y)) // 2
                padding_right = MAX_LENGTH - len(y) - padding_left

                # Pad the array on both sides
                y = np.pad(y, (padding_left, padding_right), mode='constant')

                meta = np.load(os.path.join(output_embedding_dir, meta_file_name))
                embedding = meta["embedding"]

                # Perform feature extraction (e.g., Mel spectrogram)
                mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
                filename2 = filename.split(".")[0]

                # Save audio data as numpy array
                np.savez(file=os.path.join(output_dir, f"{filename2}.npz"),
                         metadata=embedding,
                         audio=y,
                         mel_spec=mel_spec)


def process_metadata(data_dir):
    extracted_text_dir = os.path.join("/".join(data_dir.split("/")[:-1]), "extracted_text")
    output_embedding_dir = os.path.join("/".join(data_dir.split("/")[:-1]), "text_embeddings")

    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith(".flac"):
                pass
            elif filename.endswith(".txt"):
                input_txt = os.path.join(root, filename)
                save_texts_by_identifier(input_txt, extracted_text_dir)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_embedding_dir):
        os.makedirs(output_embedding_dir)

    # Initialize the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create and save embeddings
    create_and_save_embeddings(extracted_text_dir, output_embedding_dir, model)
    print("Embedding Saved!")


if __name__ == "__main__":
    # Example usage
    data_dir = "../data/librispeech/LibriSpeech/dev-clean"
    process_data(data_dir)
