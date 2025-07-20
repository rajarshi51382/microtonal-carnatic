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

def generate_gamaka_pitch_contour(base_freq, duration, sample_rate, gamaka_type='none', intensity=0.05):
    """
    Generates a pitch contour for a Swara, optionally with a simple Gamaka.

    Args:
        base_freq (float): The base frequency of the Swara.
        duration (float): The duration of the Swara in seconds.
        sample_rate (int): The sample rate of the audio.
        gamaka_type (str): Type of gamaka ('none', 'oscillation', 'glide_up', 'glide_down').
        intensity (float): Intensity of the gamaka (e.g., amplitude for oscillation, range for glide).

    Returns:
        np.ndarray: An array of frequencies over time for the Swara's duration.
    """
    num_samples = int(duration * sample_rate)
    t = np.linspace(0., duration, num_samples, endpoint=False)
    pitch_contour = np.full(num_samples, base_freq)

    if gamaka_type == 'oscillation':
        # Simple sine wave oscillation around the base frequency
        oscillation_freq = 5  # Hz, how many oscillations per second
        amplitude = base_freq * intensity
        pitch_contour += amplitude * np.sin(2 * np.pi * oscillation_freq * t)
    elif gamaka_type == 'glide_up':
        # Glide up to a slightly higher frequency and then back to base
        peak_freq = base_freq * (1 + intensity)
        mid_point = num_samples // 2
        pitch_contour[:mid_point] = np.linspace(base_freq, peak_freq, mid_point)
        pitch_contour[mid_point:] = np.linspace(peak_freq, base_freq, num_samples - mid_point)
    elif gamaka_type == 'glide_down':
        # Glide down to a slightly lower frequency and then back to base
        trough_freq = base_freq * (1 - intensity)
        mid_point = num_samples // 2
        pitch_contour[:mid_point] = np.linspace(base_freq, trough_freq, mid_point)
        pitch_contour[mid_point:] = np.linspace(trough_freq, base_freq, num_samples - mid_point)

    return pitch_contour

def synthesize_swara_sequence(swara_sequence, output_path, duration_per_swara=0.5, sample_rate=44100, portamento=0.05, gamaka_type='none', gamaka_intensity=0.05):
    """
    Synthesizes a Swara sequence into an audio file using numpy and scipy,
    with additive synthesis and optional Gamakas.

    Args:
        swara_sequence (list): A list of Swaras (e.g., ['S', 'R1', 'G3']).
        output_path (str): The path to save the output audio file (e.g., 'output.wav').
        duration_per_swara (float): The duration of each Swara in seconds.
        sample_rate (int): The sample rate of the audio.
        portamento (float): The duration of the portamento slide in seconds.
        gamaka_type (str): Type of gamaka to apply to each note ('none', 'oscillation', 'glide_up', 'glide_down').
        gamaka_intensity (float): Intensity of the gamaka.
    """
    audio_data = []
    last_freq = None

    # Additive synthesis parameters (harmonics)
    harmonics = {1: 1.0, 2: 0.5, 3: 0.25, 4: 0.125}  # Harmonic: Amplitude

    for i, swara in enumerate(swara_sequence):
        if swara in SWARA_FREQUENCIES:
            base_freq = SWARA_FREQUENCIES[swara]
            
            # Generate pitch contour for the current swara, including gamaka
            current_gamaka_type = gamaka_type
            # For demonstration, apply different gamakas to different notes
            if gamaka_type == 'auto':
                if i % 3 == 0: current_gamaka_type = 'oscillation'
                elif i % 3 == 1: current_gamaka_type = 'glide_up'
                else: current_gamaka_type = 'none'

            pitch_contour = generate_gamaka_pitch_contour(base_freq, duration_per_swara, sample_rate, current_gamaka_type, gamaka_intensity)
            
            num_samples = len(pitch_contour)
            t = np.linspace(0., duration_per_swara, num_samples, endpoint=False)
            wave = np.zeros(num_samples)

            # Apply portamento from last note's frequency to current note's starting frequency
            if last_freq is not None and portamento > 0:
                slide_samples = int(portamento * sample_rate)
                if slide_samples > 0 and slide_samples < num_samples:
                    # Create a slide from last_freq to the first frequency of the current pitch_contour
                    slide_segment = np.linspace(last_freq, pitch_contour[0], slide_samples)
                    # Replace the beginning of the pitch_contour with the slide segment
                    pitch_contour[:slide_samples] = slide_segment

            # Additive synthesis using the pitch contour
            for h, amp in harmonics.items():
                wave += amp * np.sin(2 * np.pi * pitch_contour * t * h)

            audio_data.append(wave)
            last_freq = pitch_contour[-1] # Last frequency of the current contour

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
    parser.add_argument("swara_sequence", type=str, nargs='+', help="A space-separated string of Swaras (e.g., 'S R1 G3 M1 P D1 N3 S').")
    parser.add_argument("output_path", type=str, help="The path to save the output audio file (e.g., 'output.wav').")
    parser.add_argument("--duration", type=float, default=0.5, help="The duration of each Swara in seconds.")
    parser.add_argument("--gamaka_type", type=str, default="none", choices=['none', 'oscillation', 'glide_up', 'glide_down', 'auto'], help="Type of gamaka to apply.")
    parser.add_argument("--gamaka_intensity", type=float, default=0.05, help="Intensity of the gamaka (e.g., 0.05 for 5% deviation).")

    args = parser.parse_args()

    swara_list = args.swara_sequence

    synthesize_swara_sequence(swara_list, args.output_path, args.duration, gamaka_type=args.gamaka_type, gamaka_intensity=args.gamaka_intensity)
