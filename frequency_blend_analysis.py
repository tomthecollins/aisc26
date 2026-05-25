"""
frequency_blend_analysis.py

Analyze and blend the frequency content of two audio files.

Features
--------
1. Computes FFT magnitude spectra for audio files A and B.
2. Extracts the top-N spectral peaks:
      (m_A_i, y_A_i)
      (m_B_i, y_B_i)

   where:
      m = magnitude
      y = frequency bin center (Hz)

3. Computes a similarity score in [0, 1]:
      ~1 => highly similar spectra
      ~0 => highly dissimilar spectra

4. Produces a blended output audio file:
      - retains phase from A
      - blends magnitude spectra from A and B

Inputs
------
filename_a      : path to wav/mp3
filename_b      : path to wav/mp3
blend_percent   : float in [0,100]
output_folder   : optional output folder

Dependencies
------------
pip install numpy scipy librosa soundfile

Usage
-----
python frequency_blend_analysis.py fileA.wav fileB.wav 25

or from Python:

analyze_and_blend(
    filename_a="A.wav",
    filename_b="B.wav",
    blend_percent=40,
    output_folder="output"
)
"""

import os
import argparse
import numpy as np
import librosa
import soundfile as sf


# ============================================================
# Utility Functions
# ============================================================

def load_audio(filepath, target_sr=None):
    """
    Load audio as mono float32 signal.

    Returns:
        y  : audio samples
        sr : sample rate
    """
    y, sr = librosa.load(filepath, sr=target_sr, mono=True)

    # Normalize
    max_abs = np.max(np.abs(y))
    if max_abs > 0:
        y = y / max_abs

    return y, sr


def match_lengths(a, b):
    """
    Truncate signals to equal length.
    """
    min_len = min(len(a), len(b))
    return a[:min_len], b[:min_len]


def compute_fft(signal, sr):
    """
    Compute FFT magnitude and phase.

    Returns:
        freqs
        magnitudes
        phases
        complex_fft
    """
    N = len(signal)

    fft_vals = np.fft.rfft(signal)

    magnitudes = np.abs(fft_vals)
    phases = np.angle(fft_vals)

    freqs = np.fft.rfftfreq(N, d=1.0 / sr)

    return freqs, magnitudes, phases, fft_vals


def top_n_peaks(freqs, magnitudes, n=20):
    """
    Return top-n magnitude peaks.

    Returns:
        List of tuples:
            (magnitude, frequency_hz)
    """
    indices = np.argsort(magnitudes)[::-1][:n]

    peaks = []
    for idx in indices:
        peaks.append((float(magnitudes[idx]), float(freqs[idx])))

    return peaks


# ============================================================
# Spectral Similarity
# ============================================================

def spectral_similarity(mag_a, mag_b):
    """
    Spectral similarity score in [0,1].

    Uses cosine similarity between normalized
    magnitude spectra.

    1 => very similar
    0 => very different
    """

    eps = 1e-12

    mag_a = mag_a / (np.linalg.norm(mag_a) + eps)
    mag_b = mag_b / (np.linalg.norm(mag_b) + eps)

    similarity = np.dot(mag_a, mag_b)

    # Clamp numerically
    similarity = float(np.clip(similarity, 0.0, 1.0))

    return similarity


# ============================================================
# Spectral Blending
# ============================================================

def blend_spectra(
    mag_a,
    mag_b,
    phase_a,
    blend_percent
):
    """
    Blend magnitudes from A and B while retaining
    phase from A.

    blend_percent:
        0   => pure A
        100 => pure B magnitudes with A phase
    """

    alpha = blend_percent / 100.0

    blended_mag = (1 - alpha) * mag_a + alpha * mag_b

    blended_fft = blended_mag * np.exp(1j * phase_a)

    return blended_fft


def reconstruct_audio(blended_fft):
    """
    Reconstruct time-domain signal from FFT.
    """
    y = np.fft.irfft(blended_fft)

    # Normalize
    max_abs = np.max(np.abs(y))
    if max_abs > 0:
        y = y / max_abs

    return y.astype(np.float32)


# ============================================================
# Output Utilities
# ============================================================

def write_analysis_file(
    filepath,
    peaks_a,
    peaks_b,
    similarity
):
    """
    Write numerical analysis to text file.
    """

    with open(filepath, "w") as f:

        f.write("=====================================\n")
        f.write("TOP 20 PEAKS - FILE A\n")
        f.write("Format: (m_A_i, y_A_i)\n")
        f.write("=====================================\n\n")

        for i, (m, y) in enumerate(peaks_a):
            f.write(
                f"{i+1:02d}: "
                f"(m_A_{i+1}={m:.6f}, "
                f"y_A_{i+1}={y:.2f} Hz)\n"
            )

        f.write("\n\n")

        f.write("=====================================\n")
        f.write("TOP 20 PEAKS - FILE B\n")
        f.write("Format: (m_B_i, y_B_i)\n")
        f.write("=====================================\n\n")

        for i, (m, y) in enumerate(peaks_b):
            f.write(
                f"{i+1:02d}: "
                f"(m_B_{i+1}={m:.6f}, "
                f"y_B_{i+1}={y:.2f} Hz)\n"
            )

        f.write("\n\n")

        f.write("=====================================\n")
        f.write("SPECTRAL SIMILARITY\n")
        f.write("=====================================\n\n")

        f.write(
            "Similarity function:\n"
            "S(A,B) = cosine_similarity(normalized_magnitude_spectra)\n\n"
        )

        f.write(f"Similarity score = {similarity:.6f}\n")


# ============================================================
# Main Pipeline
# ============================================================

def analyze_and_blend(
    filename_a,
    filename_b,
    blend_percent,
    output_folder="output"
):
    """
    Main processing pipeline.
    """

    os.makedirs(output_folder, exist_ok=True)

    print("Loading audio files...")

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    y_a, sr_a = load_audio(filename_a)
    y_b, sr_b = load_audio(filename_b, target_sr=sr_a)

    # Match lengths
    y_a, y_b = match_lengths(y_a, y_b)

    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    print("Computing FFTs...")

    freqs_a, mag_a, phase_a, fft_a = compute_fft(y_a, sr_a)
    freqs_b, mag_b, phase_b, fft_b = compute_fft(y_b, sr_a)

    # --------------------------------------------------------
    # Top Peaks
    # --------------------------------------------------------

    peaks_a = top_n_peaks(freqs_a, mag_a, n=20)
    peaks_b = top_n_peaks(freqs_b, mag_b, n=20)

    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    similarity = spectral_similarity(mag_a, mag_b)

    # --------------------------------------------------------
    # Blend
    # --------------------------------------------------------

    print(f"Blending spectra ({blend_percent:.1f}% B)...")

    blended_fft = blend_spectra(
        mag_a,
        mag_b,
        phase_a,
        blend_percent
    )

    blended_audio = reconstruct_audio(blended_fft)

    # --------------------------------------------------------
    # Write outputs
    # --------------------------------------------------------

    analysis_path = os.path.join(
        output_folder,
        "frequency_analysis.txt"
    )

    blended_path = os.path.join(
        output_folder,
        f"blended_{blend_percent:.0f}pct.wav"
    )

    write_analysis_file(
        analysis_path,
        peaks_a,
        peaks_b,
        similarity
    )

    sf.write(blended_path, blended_audio, sr_a)

    print("\nDone.")
    print(f"Analysis written to: {analysis_path}")
    print(f"Blended audio written to: {blended_path}")
    print(f"Similarity score: {similarity:.6f}")

    return {
        "similarity": similarity,
        "analysis_file": analysis_path,
        "blended_audio": blended_path,
    }


# ============================================================
# Command Line Interface
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Frequency analysis and blending"
    )

    parser.add_argument(
        "filename_a",
        type=str,
        help="Path to audio file A"
    )

    parser.add_argument(
        "filename_b",
        type=str,
        help="Path to audio file B"
    )

    parser.add_argument(
        "blend_percent",
        type=float,
        help="Blend amount from B into A (0-100)"
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="output",
        help="Folder for output files"
    )

    args = parser.parse_args()

    analyze_and_blend(
        filename_a=args.filename_a,
        filename_b=args.filename_b,
        blend_percent=args.blend_percent,
        output_folder=args.output_folder
    )
