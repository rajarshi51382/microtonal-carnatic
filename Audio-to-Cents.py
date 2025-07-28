# Import required libraries
import os
import librosa
import numpy as np

# Define the path to the Downloads folder (modify if your Downloads folder is elsewhere)
downloads_folder = os.path.expanduser("~/Downloads")
audio_file_name = "cents_output.wav"  # Replace with your audio file name
audio_path = os.path.join(downloads_folder, audio_file_name)

# Check if the audio file exists
if not os.path.exists(audio_path):
    print(f"Error: Audio file {audio_path} not found in Downloads folder.")
else:
    # Load the audio file
    y, sr = librosa.load(audio_path)

    # Extract pitch (fundamental frequency) using librosa's piptrack
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # Get the index of the maximum magnitude for each time frame
    pitch_values = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        if pitch > 0:  # Only include valid pitch values
            pitch_values.append(pitch)

    # Convert pitch values (Hz) to cents relative to A4 (440 Hz)
    reference_freq = 440.0  # A4 = 440 Hz
    cents = []
    for freq in pitch_values:
        if freq > 0:  # Avoid log of zero or negative frequencies
            cents_value = 1200 * np.log2(freq / reference_freq)
            cents.append(cents_value)

    # Print the sequence of cents
    print("Sequence of cents (relative to A4 = 440 Hz):")
    print(cents)

    # Optional: Save the cents sequence to a text file in Downloads
    output_path = os.path.join(downloads_folder, "cents_sequence.txt")
    np.savetxt(output_path, cents, fmt="%.2f", header="Cents relative to A4 (440 Hz)")

    print(f"Cents sequence saved to {output_path}")
