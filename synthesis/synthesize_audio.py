import argparse
import numpy as np
from scipy.io import wavfile
import time
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling.shrutisense import ShrutiUtils

# This dictionary maps Swaras to their base frequencies (in Hz).
# It is now generated dynamically from the ShrutiUtils class.
# Tonic is C4 = 261.63 Hz
BASE_FREQ = 261.63
SWARA_FREQUENCIES = {
    shruti: BASE_FREQ * (2**(cents / 1200))
    for shruti, cents in ShrutiUtils.SHRUTI_CENTS.items()
}

def synthesize_swara_sequence(swara_sequence, output_path, duration_per_swara=0.5, sample_rate=44100, portamento=0.1):
    """
    Synthesizes a Swara sequence into an audio file using numpy and scipy,
    with additive synthesis and portamento.

    Args:
        swara_sequence (list): A list of Swaras (e.g., ['S', 'R1', 'G3']).
        output_path (str): The path to save the output audio file (e.g., 'output.wav').
        duration_per_swara (float): The duration of each Swara in seconds.
        sample_rate (int): The sample rate of the audio.
        portamento (float): The duration of the portamento slide in seconds.
    """
    audio_data = []
    last_freq = None

    # Additive synthesis parameters (harmonics)
    harmonics = {1: 1.0, 2: 0.5, 3: 0.25, 4: 0.125}  # Harmonic: Amplitude

    for swara in swara_sequence:
        if swara in SWARA_FREQUENCIES:
            freq = SWARA_FREQUENCIES[swara]
            t = np.linspace(0., duration_per_swara, int(duration_per_swara * sample_rate), endpoint=False)
            wave = np.zeros_like(t)

            # Additive synthesis
            for h, amp in harmonics.items():
                wave += amp * np.sin(2 * np.pi * (freq * h) * t)

            # Apply portamento (slide)
            if last_freq is not None and portamento > 0:
                slide_samples = int(portamento * sample_rate)
                if slide_samples > 0:
                    slide = np.linspace(last_freq, freq, slide_samples)
                    wave[:slide_samples] = np.sin(2 * np.pi * slide * t[:slide_samples])

            audio_data.append(wave)
            last_freq = freq

    # Concatenate all the waves
    if audio_data:
        audio_data = np.concatenate(audio_data)
        # Normalize to 16-bit range
        audio_data = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
        wavfile.write(output_path, sample_rate, audio_data)
        print(f"Audio synthesized and saved to {output_path}")
    else:
        print("No valid swaras found in the sequence. No audio generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesizes a Swara sequence into an audio file.")
    parser.add_argument("swara_sequence", type=str, help="A space-separated string of Swaras (e.g., 'S R1 G3 M1 P D1 N3 S').")
    parser.add_argument("output_path", type=str, help="The path to save the output audio file (e.g., 'output.wav').")
    parser.add_argument("--duration", type=float, default=0.5, help="The duration of each Swara in seconds.")

    args = parser.parse_args()

    swara_list = args.swara_sequence.split()

    synthesize_swara_sequence(swara_list, args.output_path, args.duration)
