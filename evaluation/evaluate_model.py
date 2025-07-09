
import numpy as np
import argparse

def load_pitch_contour(file_path):
    """
    Loads a pitch contour from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        np.ndarray: The frequency data from the pitch contour.
    """
    try:
        # Assuming the second column is the frequency
        data = np.loadtxt(file_path, delimiter=',', skiprows=1, usecols=1)
        return data
    except Exception as e:
        print(f"Error loading pitch contour from {file_path}: {e}")
        return None

def compare_pitch_contours(generated_contour, reference_contour):
    """
    Compares two pitch contours and calculates the root mean squared error (RMSE).

    Args:
        generated_contour (np.ndarray): The pitch contour from the generated audio.
        reference_contour (np.ndarray): The pitch contour from the reference audio.

    Returns:
        float: The RMSE between the two contours.
    """
    # Pad the shorter contour to match the length of the longer one
    len_g = len(generated_contour)
    len_r = len(reference_contour)

    if len_g > len_r:
        reference_contour = np.pad(reference_contour, (0, len_g - len_r), 'constant')
    elif len_r > len_g:
        generated_contour = np.pad(generated_contour, (0, len_r - len_g), 'constant')

    # Calculate RMSE
    rmse = np.sqrt(np.mean((generated_contour - reference_contour) ** 2))
    return rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluates a generated audio file against a reference audio file.")
    parser.add_argument("generated_pitch_path", type=str, help="Path to the pitch contour file of the generated audio.")
    parser.add_argument("reference_pitch_path", type=str, help="Path to the pitch contour file of the reference audio.")

    args = parser.parse_args()

    generated_contour = load_pitch_contour(args.generated_pitch_path)
    reference_contour = load_pitch_contour(args.reference_pitch_path)

    if generated_contour is not None and reference_contour is not None:
        rmse = compare_pitch_contours(generated_contour, reference_contour)
        print(f"RMSE between the generated and reference pitch contours: {rmse:.2f} Hz")
