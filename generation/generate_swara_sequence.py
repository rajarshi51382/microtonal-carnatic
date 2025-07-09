
import numpy as np
import argparse

def generate_swara_sequence(transition_matrix, swaras, length, start_swara=None):
    """
    Generates a sequence of Swaras using a Markov chain.

    Args:
        transition_matrix (np.ndarray): The transition matrix for the Swaras.
        swaras (list): The list of Swaras corresponding to the transition matrix.
        length (int): The desired length of the Swara sequence.
        start_swara (str, optional): The starting Swara for the sequence. Defaults to a random Swara.

    Returns:
        list: The generated sequence of Swaras.
    """
    if start_swara is None:
        current_swara_index = np.random.choice(len(swaras))
    else:
        current_swara_index = swaras.index(start_swara)

    sequence = [swaras[current_swara_index]]

    for _ in range(length - 1):
        next_swara_index = np.random.choice(
            len(swaras),
            p=transition_matrix[current_swara_index]
        )
        sequence.append(swaras[next_swara_index])
        current_swara_index = next_swara_index

    return sequence

if __name__ == "__main__":
    # Example for a simple Raga (e.g., Mayamalavagowla)
    # S R1 G3 M1 P D1 N3 S
    swaras = ['S', 'R1', 'G3', 'M1', 'P', 'D1', 'N3']
    
    # A simple, placeholder transition matrix (should be learned from data)
    # This matrix is uniform, which will produce random sequences.
    # A real implementation should enforce Arohana/Avarohana and other rules.
    transition_matrix = np.array([
        [0.1, 0.5, 0.1, 0.1, 0.1, 0.1, 0.0],  # From S
        [0.1, 0.1, 0.5, 0.1, 0.1, 0.1, 0.0],  # From R1
        [0.0, 0.1, 0.1, 0.5, 0.1, 0.1, 0.1],  # From G3
        [0.1, 0.0, 0.1, 0.1, 0.5, 0.1, 0.1],  # From M1
        [0.1, 0.1, 0.0, 0.1, 0.1, 0.5, 0.1],  # From P
        [0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.5],  # From D1
        [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]   # From N3
    ])
    # Normalize rows to sum to 1
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

    parser = argparse.ArgumentParser(description="Generates a Swara sequence for a given Raga.")
    parser.add_argument("--length", type=int, default=20, help="The length of the Swara sequence to generate.")
    parser.add_argument("--start_swara", type=str, choices=swaras, help="The starting Swara of the sequence.")

    args = parser.parse_args()

    swara_sequence = generate_swara_sequence(
        transition_matrix,
        swaras,
        args.length,
        args.start_swara
    )

    print("Generated Swara Sequence:")
    print(" ".join(swara_sequence))
