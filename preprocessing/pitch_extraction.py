
import librosa
import numpy as np
import argparse

def extract_pitch_contour(audio_path):
    """
    Extracts the pitch contour from an audio file using librosa's piptrack.

    Args:
        audio_path (str): Path to the input audio file.

    Returns:
        tuple: A tuple containing:
            - times (np.ndarray): The time instances of the pitch estimates.
            - frequencies (np.ndarray): The corresponding frequency estimates.
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        # Select the pitch with the highest magnitude for each frame
        selected_pitches = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                selected_pitches.append(pitch)
            else:
                selected_pitches.append(0) # or np.nan

        times = librosa.times_like(pitches[0])
        return times, np.array(selected_pitches)

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, None

def save_pitch_contour(times, frequencies, output_path):
    """
    Saves the pitch contour to a CSV file.

    Args:
        times (np.ndarray): The time instances of the pitch estimates.
        frequencies (np.ndarray): The corresponding frequency estimates.
        output_path (str): Path to the output CSV file.
    """
    try:
        data = np.vstack((times, frequencies)).T
        np.savetxt(output_path, data, delimiter=',', header='time,frequency', comments='')
        print(f"Pitch contour saved to {output_path}")
    except Exception as e:
        print(f"Error saving to {output_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts and saves the pitch contour from an audio file.")
    parser.add_argument("audio_path", type=str, help="Path to the input audio file.")
    parser.add_argument("output_path", type=str, help="Path to the output CSV file for the pitch contour.")
    
    args = parser.parse_args()

    times, frequencies = extract_pitch_contour(args.audio_path)

    if times is not None and frequencies is not None:
        save_pitch_contour(times, frequencies, args.output_path)
