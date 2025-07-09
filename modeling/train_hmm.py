
import numpy as np
from hmmlearn import hmm
import argparse
import joblib

def load_pitch_data(data_path):
    """
    Loads pitch contour data from a CSV file.

    Args:
        data_path (str): Path to the input CSV file.

    Returns:
        np.ndarray: The pitch contour data as a numpy array.
    """
    try:
        # Assuming the second column is the frequency
        data = np.loadtxt(data_path, delimiter=',', skiprows=1, usecols=1)
        # The data needs to be in the shape (n_samples, n_features)
        return data.reshape(-1, 1)
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
        return None

def train_hmm(data, n_components=5, n_iter=100):
    """
    Trains a Gaussian Hidden Markov Model on the pitch data.

    Args:
        data (np.ndarray): The input pitch data.
        n_components (int): The number of hidden states in the HMM.
        n_iter (int): The number of iterations to train the HMM.

    Returns:
        hmm.GaussianHMM: The trained HMM model.
    """
    try:
        model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=n_iter)
        model.fit(data)
        return model
    except Exception as e:
        print(f"Error training HMM: {e}")
        return None

def save_hmm_model(model, model_path):
    """
    Saves the trained HMM model to a file.

    Args:
        model (hmm.GaussianHMM): The trained HMM model.
        model_path (str): Path to the output model file.
    """
    try:
        joblib.dump(model, model_path)
        print(f"HMM model saved to {model_path}")
    except Exception as e:
        print(f"Error saving model to {model_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a Gaussian HMM on pitch contour data.")
    parser.add_argument("data_path", type=str, help="Path to the input CSV file containing pitch data.")
    parser.add_argument("model_path", type=str, help="Path to save the trained HMM model.")
    parser.add_argument("--n_components", type=int, default=5, help="Number of hidden states in the HMM.")
    parser.add_argument("--n_iter", type=int, default=100, help="Number of iterations for training.")

    args = parser.parse_args()

    pitch_data = load_pitch_data(args.data_path)

    if pitch_data is not None:
        hmm_model = train_hmm(pitch_data, args.n_components, args.n_iter)
        if hmm_model is not None:
            save_hmm_model(hmm_model, args.model_path)
