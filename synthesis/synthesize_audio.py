
import argparse
from pyo import *
import time

# This dictionary maps Swaras to their base frequencies (in Hz).
# This is a simplified mapping and can be expanded.
# Tonic is C4 = 261.63 Hz
SWARA_FREQUENCIES = {
    'S': 261.63,
    'R1': 277.18,  # Shuddha Rishabham
    'R2': 293.66,  # Chatusruti Rishabham
    'G2': 311.13,  # Sadharana Gandharam
    'G3': 329.63,  # Antara Gandharam
    'M1': 349.23,  # Shuddha Madhyamam
    'M2': 369.99,  # Prati Madhyamam
    'P': 392.00,   # Panchamam
    'D1': 415.30,  # Shuddha Dhaivatam
    'D2': 440.00,  # Chatusruti Dhaivatam
    'N2': 466.16,  # Kaisiki Nishadam
    'N3': 493.88,  # Kakali Nishadam
}

def synthesize_swara_sequence(swara_sequence, output_path, duration_per_swara=0.5):
    """
    Synthesizes a Swara sequence into an audio file using pyo.

    Args:
        swara_sequence (list): A list of Swaras (e.g., ['S', 'R1', 'G3']).
        output_path (str): The path to save the output audio file (e.g., 'output.wav').
        duration_per_swara (float): The duration of each Swara in seconds.
    """
    s = Server(audio="offline").boot()
    s.setVerbosity(0)

    # Set the recording file
    s.recordOptions(dur=len(swara_sequence) * duration_per_swara, filename=output_path, fileformat=0, sampletype=0)

    # Create a simple sine wave oscillator
    osc = Sine(freq=261.63, mul=0.5) # Start with the base frequency of S

    # A list to hold the playback events
    events = []

    for i, swara in enumerate(swara_sequence):
        if swara in SWARA_FREQUENCIES:
            freq = SWARA_FREQUENCIES[swara]
            # Schedule the frequency change
            events.append(Call(osc.setFreq, [freq], time=i * duration_per_swara))

    # Create a Pyo Patcher to play the events
    pat = Patcher(events).out()

    # Start the server and rendering
    s.start()
    s.shutdown()

    print(f"Audio synthesized and saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesizes a Swara sequence into an audio file.")
    parser.add_argument("swara_sequence", type=str, help="A space-separated string of Swaras (e.g., 'S R1 G3 M1 P D1 N3 S').")
    parser.add_argument("output_path", type=str, help="The path to save the output audio file (e.g., 'output.wav').")
    parser.add_argument("--duration", type=float, default=0.5, help="The duration of each Swara in seconds.")

    args = parser.parse_args()

    swara_list = args.swara_sequence.split()

    synthesize_swara_sequence(swara_list, args.output_path, args.duration)
