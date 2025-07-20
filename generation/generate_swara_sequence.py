
import numpy as np
import argparse
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling.shrutisense import RagaGrammar, ShrutiUtils

def generate_swara_sequence(raga_name, length, start_swara=None):
    """
    Generates a sequence of Swaras for a given Raga using its grammar.

    Args:
        raga_name (str): The name of the Raga to generate the sequence for.
        length (int): The desired length of the Swara sequence.
        start_swara (str, optional): The starting Swara for the sequence. Defaults to a random Swara from the Raga's arohana.

    Returns:
        list: The generated sequence of Swaras.
    """
    grammar = RagaGrammar(raga_name)
    
    # Use the active shrutis from the RagaGrammar for the possible notes
    # This ensures we only generate notes valid for the given Raga
    swaras = sorted(list(set(grammar.arohana + grammar.avarohana)), key=lambda x: ShrutiUtils.shruti_to_cents(x))
    
    if not swaras:
        raise ValueError(f"No active Swaras found for Raga: {raga_name}")

    if start_swara is None:
        # Start with Sa or a random note from arohana if Sa is not available
        if 'Sa' in swaras:
            current_swara = 'Sa'
        else:
            current_swara = np.random.choice(grammar.arohana)
    else:
        if start_swara not in swaras:
            raise ValueError(f"Start Swara '{start_swara}' is not valid for Raga '{raga_name}'.")
        current_swara = start_swara

    sequence = [current_swara]

    for _ in range(length - 1):
        possible_next_swaras = []
        probabilities = []

        # Determine melodic direction for transition
        # Simple heuristic: if current note is lower than previous, assume ascending intent
        # This can be made more sophisticated
        ascending_intent = True
        if len(sequence) > 1:
            prev_shruti_cents = ShrutiUtils.shruti_to_cents(sequence[-2])
            current_shruti_cents = ShrutiUtils.shruti_to_cents(current_swara)
            if current_shruti_cents < prev_shruti_cents:
                ascending_intent = False # Actually descending

        for next_swara in swaras:
            prob = grammar.get_transition_probability(current_swara, next_swara, ascending=ascending_intent)
            if prob > 0:
                possible_next_swaras.append(next_swara)
                probabilities.append(prob)
        
        if not possible_next_swaras:
            # Fallback: if no valid transitions, pick a random note from the raga's scale
            current_swara = np.random.choice(swaras)
            sequence.append(current_swara)
            continue

        # Normalize probabilities
        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()

        current_swara = np.random.choice(possible_next_swaras, p=probabilities)
        sequence.append(current_swara)

    return sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates a Swara sequence for a given Raga.")
    parser.add_argument("--raga", type=str, default="Yaman", help="The name of the Raga (e.g., 'Yaman', 'Bhairavi').")
    parser.add_argument("--length", type=int, default=20, help="The length of the Swara sequence to generate.")
    parser.add_argument("--start_swara", type=str, help="The starting Swara of the sequence.")

    args = parser.parse_args()

    try:
        swara_sequence = generate_swara_sequence(
            args.raga,
            args.length,
            args.start_swara
        )

        print(f"Generated Swara Sequence for Raga {args.raga}:")
        print(" ".join(swara_sequence))
    except ValueError as e:
        print(f"Error: {e}")

