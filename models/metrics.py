import numpy as np
from scipy import signal
from scipy.signal import stft
from scipy.linalg import sqrtm
from scipy.io import wavfile
import librosa


def pesq(original_audio, degraded_audio, sample_rate=48000):
    # Ensure both audios have the same length
    min_length = min(len(original_audio), len(degraded_audio))
    original_audio = original_audio[:min_length]
    degraded_audio = degraded_audio[:min_length]

    # Filter audios to speech frequencies (300 Hz - 3.4 kHz)
    min_freq = 300
    max_freq = 3400
    nyquist = sample_rate / 2
    filter_order = 4
    sos = signal.butter(filter_order, [min_freq / nyquist, max_freq / nyquist], btype='band', output='sos')
    original_audio_filtered = signal.sosfilt(sos, original_audio)
    degraded_audio_filtered = signal.sosfilt(sos, degraded_audio)

    # Compute time-domain segmental SNR (log scale)
    segment_length = 128
    num_segments = len(original_audio) // segment_length
    seg_snr = np.zeros(num_segments)
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        orig_seg = original_audio_filtered[start:end]
        degr_seg = degraded_audio_filtered[start:end]
        seg_snr[i] = 10 * np.log10(np.sum(orig_seg ** 2) / np.sum((orig_seg - degr_seg) ** 2))

    # Apply frequency-domain weighting
    seg_snr_weighted = seg_snr
    # Example: seg_snr_weighted = apply_frequency_weighting(seg_snr)

    # Calculate the PESQ score (mean segmental SNR)
    pesq_score = np.mean(seg_snr_weighted)
    return pesq_score


def stoi(original_audio, degraded_audio, sample_rate=48000, window_length=256, overlap=128):
    # Ensure both audios have the same length
    min_length = min(len(original_audio), len(degraded_audio))
    original_audio = original_audio[:min_length]
    degraded_audio = degraded_audio[:min_length]

    # Perform short-time Fourier transform (STFT) on both audios
    f, t, X_orig = stft(original_audio, fs=sample_rate, window='hann', nperseg=window_length, noverlap=overlap)
    _, _, X_degr = stft(degraded_audio, fs=sample_rate, window='hann', nperseg=window_length, noverlap=overlap)

    # Compute the magnitude spectrum of the STFT
    S_orig = np.abs(X_orig)
    S_degr = np.abs(X_degr)

    # Calculate the coherence between the original and degraded signals
    coherence = np.abs(np.sum(S_orig * S_degr.conj(), axis=1)) / (
            np.sqrt(np.sum(S_orig ** 2, axis=1)) * np.sqrt(np.sum(S_degr ** 2, axis=1)))

    # Calculate STOI score
    stoi_score = np.mean(coherence)
    return stoi_score


def warpq(original_audio, degraded_audio, sample_rate=48000, window_length=256, overlap=128):
    # Ensure both audios have the same length
    min_length = min(len(original_audio), len(degraded_audio))
    original_audio = original_audio[:min_length]
    degraded_audio = degraded_audio[:min_length]

    # Perform short-time Fourier transform (STFT) on both audios
    f, t, X_orig = stft(original_audio, fs=sample_rate, window='hann', nperseg=window_length, noverlap=overlap)
    _, _, X_degr = stft(degraded_audio, fs=sample_rate, window='hann', nperseg=window_length, noverlap=overlap)

    # Compute the relative phase between original and degraded signals
    phase_orig = np.angle(X_orig)
    phase_degr = np.angle(X_degr)
    relative_phase = np.angle(np.exp(1j * (phase_degr - phase_orig)))

    # Calculate WARP-Q score
    num_freq_bins = relative_phase.shape[0]
    weights = np.arange(1, num_freq_bins + 1) / num_freq_bins  # Weighted by frequency bin index
    warpq_score = np.sum(np.abs(relative_phase.reshape(-1, 1)) * weights) / np.sum(weights)
    return warpq_score


def fad(original_audio, degraded_audio, sample_rate=48000, window_length=256, overlap=128):
    # Ensure both audios have the same length
    min_length = min(len(original_audio), len(degraded_audio))
    original_audio = original_audio[:min_length]
    degraded_audio = degraded_audio[:min_length]

    # Perform short-time Fourier transform (STFT) on both audios
    _, _, X_orig = stft(original_audio, fs=sample_rate, window='hann', nperseg=window_length, noverlap=overlap)
    _, _, X_degr = stft(degraded_audio, fs=sample_rate, window='hann', nperseg=window_length, noverlap=overlap)

    # Compute the covariance matrices of the STFTs
    cov_orig = np.cov(np.abs(X_orig), rowvar=False)
    cov_degr = np.cov(np.abs(X_degr), rowvar=False)

    # Compute the squared root of the product of the covariance matrices
    sqrt_cov_prod = sqrtm(cov_orig @ cov_degr)

    # Compute the Frechet distance
    fad_score = np.sqrt(np.trace(cov_orig) + np.trace(cov_degr) - 2 * np.trace(sqrt_cov_prod))
    return fad_score


def compute_mfcc(audio, sr=22050, n_mfcc=13):
    # Compute MFCCs for the audio signal
    mfcc = librosa.feature.mfcc(y=audio.flatten(), sr=sr, n_mfcc=n_mfcc)
    return mfcc


def mcd13(audio1, audio2, sr=22050, n_mfcc=13):
    # Compute MFCCs for both audio signals
    mfcc1 = compute_mfcc(audio1, sr=sr, n_mfcc=n_mfcc)
    mfcc2 = compute_mfcc(audio2, sr=sr, n_mfcc=n_mfcc)

    # Calculate Mel-Cepstral Distortion (MCD13)
    mcd13 = np.mean(np.sqrt(2 * np.sum((mfcc1 - mfcc2) ** 2, axis=0)))
    return mcd13


def rmse_fo(audio1, audio2, sr=22050, n_fft=2048):
    # Compute STFT (Short-Time Fourier Transform) for both audio signals
    stft1 = librosa.stft(audio1.flatten(), n_fft=n_fft)
    stft2 = librosa.stft(audio2.flatten(), n_fft=n_fft)

    # Compute magnitude spectrogram
    mag_spec1 = np.abs(stft1)
    mag_spec2 = np.abs(stft2)

    # Compute RMSE_fo
    rmse_fo = np.sqrt(np.mean((20 * np.log10(mag_spec1) - 20 * np.log10(mag_spec2)) ** 2))
    return rmse_fo


def fdsd(audio1, audio2):
    # Compute FFT (Fast Fourier Transform) for both audio signals
    fft1 = np.fft.fft(audio1.flatten())
    fft2 = np.fft.fft(audio2.flatten())

    # Compute frequency response
    freq_resp1 = np.abs(fft1)
    freq_resp2 = np.abs(fft2)

    # Compute FDSD
    fdsd = np.mean(np.abs(freq_resp1 - freq_resp2))
    return fdsd


def calculate_metrics(generated_audio_path, real_audio_path):
    generated_rate, generated_audio = wavfile.read(generated_audio_path)
    real_rate, real_audio = wavfile.read(real_audio_path)

    generated_audio = signal.resample(generated_audio, len(real_audio))

    print(real_audio.shape, generated_audio.shape)

    # Compute PESQ score
    pesq_score = pesq(real_audio, generated_audio)
    print("PESQ score:", pesq_score)

    # Compute STOI score
    stoi_score = stoi(real_audio, generated_audio)
    print("STOI score:", stoi_score)

    # Compute WARP-Q score
    warpq_score = warpq(real_audio, generated_audio)
    print("WARP-Q score:", warpq_score)

    # Compute FAD score
    fad_score = fad(real_audio, generated_audio)
    print("FAD score:", fad_score)

    # Compute FDSD score
    fdsd_score = fdsd(real_audio, generated_audio)
    print("FDSD score:", fdsd_score)


if __name__ == "__main__":
    # Example usage:
    # Load your original and degraded audio files
    generated_audio_path = '../data/ljspeech/LJ001-0001.wav'
    real_audio_path = '../data/ljspeech/LJ001-0002.wav'

    calculate_metrics(generated_audio_path, real_audio_path)
