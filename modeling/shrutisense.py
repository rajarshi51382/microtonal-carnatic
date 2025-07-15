"""
ShrutiSense: Symbolic Music Correction Tool for Cultural Restoration

A complete Python implementation of a microtonal music correction pipeline
for symbolic Indian classical music, centered on Shruti recovery.

This module implements:
1. Grammar-Constrained Shruti Hidden Markov Model (GC-SHMM)
2. Shruti-based Finite-State Transducer (FST)
"""

import numpy as np
from scipy.stats import norm
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import networkx as nx
from collections import defaultdict
import json


@dataclass
class CorrectionResult:
    """Data structure for correction results"""
    label: str
    pitch: float
    error: float
    confidence: float = 0.0


class ShrutiUtils:
    """Utility class for Shruti-related operations and conversions"""

    # Complete 22-Shruti system with approximate cent values
    SHRUTI_CENTS = {
        'Sa': 0,      # Shadja
        'Re1': 90,    # Suddha Rishabh
        'Re2': 182,   # Chatushruti Rishabh
        'Ga1': 294,   # Suddha Gandhar
        'Ga2': 386,   # Antara Gandhar
        'Ma1': 498,   # Suddha Madhyam
        'Ma2': 590,   # Prati Madhyam
        'Pa': 702,    # Pancham
        'Dha1': 792,  # Suddha Dhaivat
        'Dha2': 884,  # Chatushruti Dhaivat
        'Ni1': 996,   # Suddha Nishad
        'Ni2': 1088,  # Kaisiki Nishad
        'Sa_': 1200,  # Octave Sa
        # Additional microtonal Shrutis
        'Re1_': 112,  # Shatshruti Rishabh
        'Re2_': 204,  # Chatushruti Rishabh variant
        'Ga1_': 316,  # Suddha Gandhar variant
        'Ga2_': 408,  # Antara Gandhar variant
        'Ma1_': 520,  # Suddha Madhyam variant
        'Ma2_': 612,  # Prati Madhyam variant
        'Dha1_': 814, # Suddha Dhaivat variant
        'Dha2_': 906, # Chatushruti Dhaivat variant
        'Ni1_': 1018, # Suddha Nishad variant
    }

    # 12-TET Western scale in cents
    WESTERN_CENTS = {i * 100: f"Note_{i}" for i in range(12)}

    @staticmethod
    def get_shruti_list() -> List[str]:
        """Get ordered list of all Shrutis"""
        return sorted(ShrutiUtils.SHRUTI_CENTS.keys(),
                     key=lambda x: ShrutiUtils.SHRUTI_CENTS[x])

    @staticmethod
    def cents_to_shruti(cents: float, tolerance: float = 50) -> Optional[str]:
        """Convert cent value to nearest Shruti"""
        min_diff = float('inf')
        best_shruti = None

        for shruti, shruti_cents in ShrutiUtils.SHRUTI_CENTS.items():
            diff = abs(cents - shruti_cents)
            if diff < min_diff and diff <= tolerance:
                min_diff = diff
                best_shruti = shruti

        return best_shruti

    @staticmethod
    def shruti_to_cents(shruti: str) -> float:
        """Convert Shruti name to cent value"""
        return ShrutiUtils.SHRUTI_CENTS.get(shruti, 0)

    @staticmethod
    def western_to_cents(western_note: int) -> float:
        """Convert Western 12-TET note to cents"""
        return western_note * 100

    @staticmethod
    def normalize_cents(cents: float) -> float:
        """Normalize cents to 0-1200 range"""
        return cents % 1200


class RagaGrammar:
    """
    Encodes raga-specific grammar constraints for transitions and structure
    """

    def __init__(self, raga_name: str):
        self.raga_name = raga_name
        self.arohana = []  # Ascending scale
        self.avarohana = []  # Descending scale
        self.vadi = None  # Primary note
        self.samvadi = None  # Secondary note
        self.varjya = []  # Forbidden notes
        self.pakad = []  # Characteristic phrases

        self._initialize_raga_rules()

    def _initialize_raga_rules(self):
        """Initialize raga-specific rules"""
        raga_definitions = {
            'Yaman': {
                'arohana': ['Sa', 'Re2', 'Ga2', 'Ma2', 'Pa', 'Dha2', 'Ni2', 'Sa_'],
                'avarohana': ['Sa_', 'Ni2', 'Dha2', 'Pa', 'Ma2', 'Ga2', 'Re2', 'Sa'],
                'vadi': 'Ga2',
                'samvadi': 'Ni2',
                'varjya': ['Ma1'],
                'pakad': [['Ni2', 'Re2', 'Ga2'], ['Ma2', 'Pa', 'Dha2']]
            },
            'Bhairavi': {
                'arohana': ['Sa', 'Re1', 'Ga1', 'Ma1', 'Pa', 'Dha1', 'Ni1', 'Sa_'],
                'avarohana': ['Sa_', 'Ni1', 'Dha1', 'Pa', 'Ma1', 'Ga1', 'Re1', 'Sa'],
                'vadi': 'Ma1',
                'samvadi': 'Sa',
                'varjya': [],
                'pakad': [['Sa', 'Re1', 'Ga1'], ['Ma1', 'Pa', 'Dha1']]
            },
            'Bilaval': {
                'arohana': ['Sa', 'Re2', 'Ga2', 'Ma1', 'Pa', 'Dha2', 'Ni2', 'Sa_'],
                'avarohana': ['Sa_', 'Ni2', 'Dha2', 'Pa', 'Ma1', 'Ga2', 'Re2', 'Sa'],
                'vadi': 'Sa',
                'samvadi': 'Pa',
                'varjya': [],
                'pakad': [['Sa', 'Re2', 'Ga2'], ['Pa', 'Dha2', 'Ni2']]
            }
        }

        if self.raga_name in raga_definitions:
            raga_def = raga_definitions[self.raga_name]
            self.arohana = raga_def['arohana']
            self.avarohana = raga_def['avarohana']
            self.vadi = raga_def['vadi']
            self.samvadi = raga_def['samvadi']
            self.varjya = raga_def['varjya']
            self.pakad = raga_def['pakad']

    def get_allowed_transitions(self, ascending: bool = True) -> Dict[str, List[str]]:
        """
        Get allowed transitions based on raga grammar

        Args:
            ascending: True for arohana, False for avarohana

        Returns:
            Dictionary mapping each Shruti to allowed next Shrutis
        """
        scale = self.arohana if ascending else self.avarohana
        transitions = defaultdict(list)

        for i in range(len(scale) - 1):
            current = scale[i]
            next_note = scale[i + 1]
            transitions[current].append(next_note)

            # Allow staying on same note
            transitions[current].append(current)

            # Allow some flexibility for ornaments
            if i < len(scale) - 2:
                transitions[current].append(scale[i + 2])

        return dict(transitions)

    def is_valid_transition(self, from_shruti: str, to_shruti: str,
                          ascending: bool = True) -> bool:
        """Check if transition is valid according to raga grammar"""
        allowed = self.get_allowed_transitions(ascending)
        return to_shruti in allowed.get(from_shruti, [])

    def get_transition_probability(self, from_shruti: str, to_shruti: str,
                                 ascending: bool = True) -> float:
        """Get transition probability based on raga grammar"""
        if not self.is_valid_transition(from_shruti, to_shruti, ascending):
            return 0.0

        # Higher probability for scale-wise movement
        scale = self.arohana if ascending else self.avarohana
        try:
            from_idx = scale.index(from_shruti)
            to_idx = scale.index(to_shruti)

            # Same note (staying)
            if from_idx == to_idx:
                return 0.3

            # Scale-wise movement
            if abs(from_idx - to_idx) == 1:
                return 0.6

            # Skip movement
            if abs(from_idx - to_idx) == 2:
                return 0.1

            return 0.05
        except ValueError:
            return 0.1


class ShrutiModel(ABC):
    """Abstract base class for Shruti correction models"""

    def __init__(self, raga: str):
        self.raga = raga
        self.grammar = RagaGrammar(raga)
        self.shrutis = ShrutiUtils.get_shruti_list()
        self.shruti_to_idx = {s: i for i, s in enumerate(self.shrutis)}
        self.idx_to_shruti = {i: s for i, s in enumerate(self.shrutis)}

    @abstractmethod
    def correct_sequence(self, westernized_sequence: List[float]) -> List[CorrectionResult]:
        """Correct a sequence of westernized pitches"""
        pass

    def evaluate_correction(self, corrected: List[CorrectionResult],
                          ground_truth: List[float]) -> Dict[str, float]:
        """
        Evaluate correction quality against ground truth

        Args:
            corrected: List of correction results
            ground_truth: True pitch values in cents

        Returns:
            Dictionary with evaluation metrics
        """
        if len(corrected) != len(ground_truth):
            raise ValueError("Corrected and ground truth sequences must have same length")

        total_error = 0
        correct_notes = 0

        for i, (corr, true_pitch) in enumerate(zip(corrected, ground_truth)):
            error = abs(corr.pitch - true_pitch)
            total_error += error

            # Note-level accuracy (within ±20 cents)
            if error <= 20:
                correct_notes += 1

        avg_error = total_error / len(corrected)
        note_accuracy = correct_notes / len(corrected)

        return {
            'avg_pitch_error': avg_error,
            'note_accuracy': note_accuracy,
            'total_notes': len(corrected)
        }


class ShrutiHMM(ShrutiModel):
    """
    Grammar-Constrained Shruti Hidden Markov Model

    Hidden states: 22 Shrutis (or raga-specific subset)
    Observations: Westernized 12-TET pitches in cents
    """

    def __init__(self, raga: str, sigma: float = 25.0):
        super().__init__(raga)
        self.sigma = sigma  # Standard deviation for Gaussian emission

        # Get raga-specific Shrutis
        self.active_shrutis = self._get_active_shrutis()
        self.n_states = len(self.active_shrutis)
        self.state_to_idx = {s: i for i, s in enumerate(self.active_shrutis)}
        self.idx_to_state = {i: s for i, s in enumerate(self.active_shrutis)}

        # Initialize HMM parameters
        self.pi = self._initialize_start_probabilities()
        self.A_up = self._initialize_transition_matrix(ascending=True)
        self.A_down = self._initialize_transition_matrix(ascending=False)
        self.B = self._initialize_emission_matrix()

    def _get_active_shrutis(self) -> List[str]:
        """Get active Shrutis for the raga"""
        # Use arohana and avarohana to determine active Shrutis
        active = set(self.grammar.arohana + self.grammar.avarohana)
        return sorted(active, key=lambda x: ShrutiUtils.shruti_to_cents(x))

    def _initialize_start_probabilities(self) -> np.ndarray:
        """Initialize start state probabilities"""
        pi = np.zeros(self.n_states)

        # Higher probability for Sa (tonic)
        if 'Sa' in self.state_to_idx:
            pi[self.state_to_idx['Sa']] = 0.5

        # Distribute remaining probability uniformly
        remaining_prob = 1.0 - pi.sum()
        for i in range(self.n_states):
            if pi[i] == 0:
                pi[i] = remaining_prob / (self.n_states - 1)

        return pi

    def _initialize_transition_matrix(self, ascending: bool = True) -> np.ndarray:
        """
        Initialize transition matrix based on raga grammar

        Args:
            ascending: True for arohana-based transitions, False for avarohana

        Returns:
            Transition matrix A[i,j] = P(state_j | state_i)
        """
        A = np.zeros((self.n_states, self.n_states))

        for i, from_shruti in enumerate(self.active_shrutis):
            for j, to_shruti in enumerate(self.active_shrutis):
                prob = self.grammar.get_transition_probability(
                    from_shruti, to_shruti, ascending
                )
                A[i, j] = prob

        # Normalize rows to ensure proper probability distribution
        for i in range(self.n_states):
            row_sum = A[i, :].sum()
            if row_sum > 0:
                A[i, :] /= row_sum
            else:
                # Uniform distribution if no transitions defined
                A[i, :] = 1.0 / self.n_states

        return A

    def _initialize_emission_matrix(self) -> np.ndarray:
        """
        Initialize emission matrix for Gaussian emissions

        Returns:
            Emission matrix B[i,o] = P(observation_o | state_i)
        """
        # For continuous observations, we'll compute emissions on-the-fly
        # This placeholder represents the Gaussian parameters
        emissions = np.zeros((self.n_states, 2))  # [mean, std]

        for i, shruti in enumerate(self.active_shrutis):
            emissions[i, 0] = ShrutiUtils.shruti_to_cents(shruti)  # mean
            emissions[i, 1] = self.sigma  # standard deviation

        return emissions

    def _compute_emission_probability(self, state_idx: int, observation: float) -> float:
        """
        Compute emission probability for Gaussian model

        P(o_t | s_t) = (1/√(2πσ²)) * exp(-((o_t - μ_s_t)²)/(2σ²))

        Args:
            state_idx: Index of the hidden state (Shruti)
            observation: Observed pitch in cents

        Returns:
            Emission probability
        """
        mean = self.B[state_idx, 0]
        std = self.B[state_idx, 1]

        return norm.pdf(observation, mean, std)

    def _detect_melodic_direction(self, observations: List[float]) -> List[bool]:
        """
        Detect melodic direction for each transition

        Args:
            observations: Sequence of pitch observations

        Returns:
            List of boolean values (True for ascending, False for descending)
        """
        directions = []

        for i in range(len(observations) - 1):
            ascending = observations[i + 1] > observations[i]
            directions.append(ascending)

        return directions

    def viterbi_decode(self, observations: List[float]) -> Tuple[List[int], float]:
        """
        Viterbi algorithm for finding most likely state sequence

        Args:
            observations: Sequence of pitch observations in cents

        Returns:
            Tuple of (state_sequence, log_probability)
        """
        T = len(observations)

        # Initialize Viterbi tables
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        # Detect melodic directions
        directions = self._detect_melodic_direction(observations)

        # Initialization (t=0)
        for i in range(self.n_states):
            emission_prob = self._compute_emission_probability(i, observations[0])
            delta[0, i] = np.log(self.pi[i]) + np.log(emission_prob + 1e-10)
            psi[0, i] = 0

        # Recursion (t=1 to T-1)
        for t in range(1, T):
            ascending = directions[t-1] if t-1 < len(directions) else True
            A = self.A_up if ascending else self.A_down

            for j in range(self.n_states):
                emission_prob = self._compute_emission_probability(j, observations[t])

                # Find best previous state
                scores = delta[t-1, :] + np.log(A[:, j] + 1e-10)
                best_prev = np.argmax(scores)

                delta[t, j] = scores[best_prev] + np.log(emission_prob + 1e-10)
                psi[t, j] = best_prev

        # Termination - find best final state
        best_final = np.argmax(delta[T-1, :])
        best_prob = delta[T-1, best_final]

        # Backtrack to find best path
        path = [0] * T
        path[T-1] = best_final

        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]

        return path, best_prob

    def correct_sequence(self, westernized_sequence: List[float]) -> List[CorrectionResult]:
        """
        Correct a sequence of westernized pitches using HMM

        Args:
            westernized_sequence: List of pitch values in cents (12-TET)

        Returns:
            List of correction results
        """
        if not westernized_sequence:
            return []

        # Run Viterbi decoding
        state_path, log_prob = self.viterbi_decode(westernized_sequence)

        # Convert states back to Shruti labels and pitches
        results = []
        for i, (state_idx, original_pitch) in enumerate(zip(state_path, westernized_sequence)):
            shruti_label = self.idx_to_state[state_idx]
            corrected_pitch = ShrutiUtils.shruti_to_cents(shruti_label)
            error = corrected_pitch - original_pitch

            # Compute confidence based on emission probability
            confidence = self._compute_emission_probability(state_idx, original_pitch)

            results.append(CorrectionResult(
                label=shruti_label,
                pitch=corrected_pitch,
                error=error,
                confidence=confidence
            ))

        return results


class ShrutiFST(ShrutiModel):
    """
    Shruti-based Finite-State Transducer

    Uses symbolic correction rules and raga grammar constraints
    """

    def __init__(self, raga: str, tolerance: float = 50.0):
        super().__init__(raga)
        self.tolerance = tolerance

        # Build FST graph
        self.fst_graph = self._build_fst_graph()

        # Current state tracking
        self.current_state = 'Sa'  # Start from tonic

    def _build_fst_graph(self) -> nx.DiGraph:
        """Build FST graph with states and transitions"""
        G = nx.DiGraph()

        # Add states (Shrutis)
        active_shrutis = set(self.grammar.arohana + self.grammar.avarohana)
        for shruti in active_shrutis:
            G.add_node(shruti, cents=ShrutiUtils.shruti_to_cents(shruti))

        # Add transitions based on raga grammar
        for from_shruti in active_shrutis:
            # Ascending transitions
            allowed_up = self.grammar.get_allowed_transitions(ascending=True)
            for to_shruti in allowed_up.get(from_shruti, []):
                if to_shruti in active_shrutis:
                    weight = self.grammar.get_transition_probability(
                        from_shruti, to_shruti, ascending=True
                    )
                    G.add_edge(from_shruti, to_shruti,
                             weight=weight, direction='up')

            # Descending transitions
            allowed_down = self.grammar.get_allowed_transitions(ascending=False)
            for to_shruti in allowed_down.get(from_shruti, []):
                if to_shruti in active_shrutis:
                    weight = self.grammar.get_transition_probability(
                        from_shruti, to_shruti, ascending=False
                    )
                    G.add_edge(from_shruti, to_shruti,
                             weight=weight, direction='down')

        return G

    def _find_nearest_shruti(self, cents: float,
                           current_state: str = None) -> Tuple[str, float]:
        """
        Find nearest valid Shruti with FST constraints

        Args:
            cents: Input pitch in cents
            current_state: Current FST state

        Returns:
            Tuple of (shruti_label, confidence_score)
        """
        candidates = []

        # If we have a current state, prefer valid transitions
        if current_state and current_state in self.fst_graph:
            neighbors = list(self.fst_graph.neighbors(current_state))

            for neighbor in neighbors:
                neighbor_cents = self.fst_graph.nodes[neighbor]['cents']
                distance = abs(cents - neighbor_cents)

                if distance <= self.tolerance:
                    edge_weight = self.fst_graph[current_state][neighbor]['weight']
                    score = edge_weight * (1.0 - distance / self.tolerance)
                    candidates.append((neighbor, score))

        # If no valid transitions found, find globally nearest
        if not candidates:
            for shruti in self.fst_graph.nodes():
                shruti_cents = self.fst_graph.nodes[shruti]['cents']
                distance = abs(cents - shruti_cents)

                if distance <= self.tolerance:
                    score = 1.0 - distance / self.tolerance
                    candidates.append((shruti, score))

        if not candidates:
            # Fallback to closest Shruti regardless of tolerance
            best_shruti = min(
                self.fst_graph.nodes(),
                key=lambda s: abs(cents - self.fst_graph.nodes[s]['cents'])
            )
            return best_shruti, 0.1

        # Return best candidate
        best_shruti, best_score = max(candidates, key=lambda x: x[1])
        return best_shruti, best_score

    def _apply_correction_rules(self, sequence: List[str]) -> List[str]:
        """
        Apply symbolic correction rules to improve sequence

        Args:
            sequence: List of Shruti labels

        Returns:
            Corrected sequence
        """
        corrected = sequence.copy()

        # Rule 1: Enforce cadence patterns
        if len(corrected) >= 2:
            # Common cadence: end phrases on strong beats
            if corrected[-1] not in ['Sa', 'Pa', self.grammar.vadi]:
                # Try to find a better ending
                for candidate in ['Sa', 'Pa', self.grammar.vadi]:
                    if candidate in self.fst_graph.nodes():
                        corrected[-1] = candidate
                        break

        # Rule 2: Smooth illegal jumps
        for i in range(len(corrected) - 1):
            current = corrected[i]
            next_note = corrected[i + 1]

            # Check if transition is valid
            if not self.grammar.is_valid_transition(current, next_note):
                # Find intermediate note
                arohana = self.grammar.arohana
                avarohana = self.grammar.avarohana

                try:
                    current_idx = arohana.index(current)
                    next_idx = arohana.index(next_note)

                    # Insert intermediate note for large jumps
                    if abs(next_idx - current_idx) > 2:
                        intermediate_idx = (current_idx + next_idx) // 2
                        if 0 <= intermediate_idx < len(arohana):
                            # This would require sequence expansion
                            # For now, just correct the next note
                            corrected[i + 1] = arohana[current_idx + 1]
                except ValueError:
                    # Notes not in arohana, keep as is
                    pass

        # Rule 3: Ensure pakad (characteristic phrases) are preserved
        # This is a simplified implementation
        pakad_phrases = self.grammar.pakad
        for phrase in pakad_phrases:
            if len(phrase) <= len(corrected):
                # Look for partial matches and complete them
                for i in range(len(corrected) - len(phrase) + 1):
                    segment = corrected[i:i + len(phrase)]
                    matches = sum(1 for a, b in zip(segment, phrase) if a == b)

                    # If mostly matches, complete the phrase
                    if matches >= len(phrase) * 0.6:
                        corrected[i:i + len(phrase)] = phrase

        return corrected

    def correct_sequence(self, westernized_sequence: List[float]) -> List[CorrectionResult]:
        """
        Correct sequence using FST with symbolic rules

        Args:
            westernized_sequence: List of pitch values in cents (12-TET)

        Returns:
            List of correction results
        """
        if not westernized_sequence:
            return []

        # Phase 1: Map each pitch to nearest Shruti
        initial_mapping = []
        current_state = 'Sa'

        for cents in westernized_sequence:
            shruti, confidence = self._find_nearest_shruti(cents, current_state)
            initial_mapping.append((shruti, confidence))
            current_state = shruti

        # Phase 2: Apply symbolic correction rules
        shruti_sequence = [mapping[0] for mapping in initial_mapping]
        corrected_sequence = self._apply_correction_rules(shruti_sequence)

        # Phase 3: Generate final results
        results = []
        for i, (original_cents, corrected_shruti) in enumerate(
            zip(westernized_sequence, corrected_sequence)
        ):
            corrected_pitch = ShrutiUtils.shruti_to_cents(corrected_shruti)
            error = corrected_pitch - original_cents

            # Use confidence from initial mapping
            confidence = initial_mapping[i][1] if i < len(initial_mapping) else 0.5

            results.append(CorrectionResult(
                label=corrected_shruti,
                pitch=corrected_pitch,
                error=error,
                confidence=confidence
            ))

        return results


def create_test_data() -> Dict[str, List[float]]:
    """Create test data for model evaluation"""

    # Yaman raga test sequence (Sa Re Ga Ma Pa Dha Ni Sa)
    ground_truth = [
        0,    # Sa
        182,  # Re2
        386,  # Ga2
        590,  # Ma2
        702,  # Pa
        884,  # Dha2
        1088, # Ni2
        1200  # Sa (octave)
    ]

    # Westernized version (snapped to 12-TET)
    westernized = [
        0,    # Sa -> C (0 cents)
        200,  # Re2 -> D (200 cents)
        400,  # Ga2 -> E (400 cents)
        600,  # Ma2 -> F# (600 cents)
        700,  # Pa -> G (700 cents)
        900,  # Dha2 -> A (900 cents)
        1100, # Ni2 -> B (1100 cents)
        1200  # Sa -> C (1200 cents)
    ]

    return {
        'ground_truth': ground_truth,
        'westernized': westernized,
        'raga': 'Yaman'
    }


def run_comparative_analysis():
    """Run comparative analysis between HMM and FST models"""

    # Create test data
    test_data = create_test_data()

    print("=== ShrutiSense: Comparative Analysis ===")
    print(f"Raga: {test_data['raga']}")
    print(f"Sequence length: {len(test_data['westernized'])}")
    print()

    # Initialize models
    hmm_model = ShrutiHMM(test_data['raga'])
    fst_model = ShrutiFST(test_data['raga'])

    # Correct sequences
    print("Correcting with HMM...")
    hmm_results = hmm_model.correct_sequence(test_data['westernized'])

    print("Correcting with FST...")
    fst_results = fst_model.correct_sequence(test_data['westernized'])

    # Evaluate results
    hmm_eval = hmm_model.evaluate_correction(hmm_results, test_data['ground_truth'])
    fst_eval = fst_model.evaluate_correction(fst_results, test_data['ground_truth'])

    # Display results
    print("\n=== HMM Results ===")
    print(f"Average pitch error: {hmm_eval['avg_pitch_error']:.2f} cents")
    print(f"Note accuracy: {hmm_eval['note_accuracy']:.2%}")
    print("\nDetailed corrections:")
    for i, result in enumerate(hmm_results):
        original = test_data['westernized'][i]
        truth = test_data['ground_truth'][i]
        print(f"  {i+1:2d}: {original:4.0f}c -> {result.label:4s} ({result.pitch:4.0f}c) "
              f"[Truth: {truth:4.0f}c, Error: {result.error:+6.1f}c, "
              f"Conf: {result.confidence:.3f}]")

    print("\n=== FST Results ===")
    print(f"Average pitch error: {fst_eval['avg_pitch_error']:.2f} cents")
    print(f"Note accuracy: {fst_eval['note_accuracy']:.2%}")
    print("\nDetailed corrections:")
    for i, result in enumerate(fst_results):
        original = test_data['westernized'][i]
        truth = test_data['ground_truth'][i]
        print(f"  {i+1:2d}: {original:4.0f}c -> {result.label:4s} ({result.pitch:4.0f}c) "
              f"[Truth: {truth:4.0f}c, Error: {result.error:+6.1f}c, "
              f"Conf: {result.confidence:.3f}]")

    # Comparative analysis
    print("\n=== Comparative Analysis ===")
    print(f"HMM vs FST - Average Error: {hmm_eval['avg_pitch_error']:.2f} vs {fst_eval['avg_pitch_error']:.2f} cents")
    print(f"HMM vs FST - Note Accuracy: {hmm_eval['note_accuracy']:.2%} vs {fst_eval['note_accuracy']:.2%}")

    better_model = "HMM" if hmm_eval['avg_pitch_error'] < fst_eval['avg_pitch_error'] else "FST"
    print(f"Better model (lower error): {better_model}")

    return {
        'hmm_results': hmm_results,
        'fst_results': fst_results,
        'hmm_eval': hmm_eval,
        'fst_eval': fst_eval
    }


def analyze_transition_matrices():
    """Analyze and visualize transition matrices for different ragas"""

    print("\n=== Transition Matrix Analysis ===")

    for raga in ['Yaman', 'Bhairavi', 'Bilaval']:
        print(f"\n--- {raga} Raga ---")

        hmm = ShrutiHMM(raga)

        print(f"Active Shrutis: {hmm.active_shrutis}")
        print(f"Arohana: {hmm.grammar.arohana}")
        print(f"Avarohana: {hmm.grammar.avarohana}")
        print(f"Vadi: {hmm.grammar.vadi}, Samvadi: {hmm.grammar.samvadi}")

        # Display transition probabilities for key transitions
        print("\nKey Ascending Transitions:")
        for i, from_shruti in enumerate(hmm.active_shrutis[:5]):  # Show first 5
            print(f"  {from_shruti}:")
            for j, to_shruti in enumerate(hmm.active_shrutis):
                prob = hmm.A_up[i, j]
                if prob > 0.1:  # Show only significant probabilities
                    print(f"    -> {to_shruti}: {prob:.3f}")


def test_melodic_direction_detection():
    """Test melodic direction detection in HMM"""

    print("\n=== Melodic Direction Detection Test ===")

    # Test sequences with different melodic patterns
    test_sequences = {
        'ascending': [0, 200, 400, 600, 800],
        'descending': [800, 600, 400, 200, 0],
        'mixed': [0, 200, 100, 400, 300, 500]
    }

    hmm = ShrutiHMM('Yaman')

    for pattern_name, sequence in test_sequences.items():
        print(f"\n{pattern_name.capitalize()} pattern: {sequence}")
        directions = hmm._detect_melodic_direction(sequence)
        direction_labels = ['↑' if d else '↓' for d in directions]
        print(f"Detected directions: {direction_labels}")


def test_fst_correction_rules():
    """Test FST symbolic correction rules"""

    print("\n=== FST Correction Rules Test ===")

    fst = ShrutiFST('Yaman')

    # Test sequences with different issues
    test_cases = [
        {
            'name': 'Illegal jump',
            'sequence': ['Sa', 'Ga2', 'Sa', 'Ni2'],  # Large jump from Sa to Ga2
            'expected_fix': 'Insert intermediate notes'
        },
        {
            'name': 'Weak ending',
            'sequence': ['Sa', 'Re2', 'Ga2', 'Re2'],  # Ends on Re2 instead of strong note
            'expected_fix': 'Change ending to Sa, Pa, or Vadi'
        },
        {
            'name': 'Broken pakad',
            'sequence': ['Ni2', 'Re2', 'Ma2'],  # Should be Ni2-Re2-Ga2
            'expected_fix': 'Complete pakad phrase'
        }
    ]

    for test_case in test_cases:
        print(f"\n{test_case['name']}:")
        print(f"  Original: {test_case['sequence']}")

        corrected = fst._apply_correction_rules(test_case['sequence'])
        print(f"  Corrected: {corrected}")
        print(f"  Expected: {test_case['expected_fix']}")

        # Check if correction was applied
        if corrected != test_case['sequence']:
            print("  ✓ Correction applied")
        else:
            print("  ✗ No correction applied")


def export_results_to_json(results: Dict, filename: str = "shrutisense_results.json"):
    """Export results to JSON format for further analysis"""

    # Convert results to JSON-serializable format
    json_results = {
        'hmm_results': [
            {
                'label': r.label,
                'pitch': r.pitch,
                'error': r.error,
                'confidence': r.confidence
            }
            for r in results['hmm_results']
        ],
        'fst_results': [
            {
                'label': r.label,
                'pitch': r.pitch,
                'error': r.error,
                'confidence': r.confidence
            }
            for r in results['fst_results']
        ],
        'hmm_evaluation': results['hmm_eval'],
        'fst_evaluation': results['fst_eval']
    }

    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults exported to {filename}")


def demonstrate_raga_specificity():
    """Demonstrate how different ragas produce different corrections"""

    print("\n=== Raga Specificity Demonstration ===")

    # Same westernized sequence corrected in different ragas
    westernized = [0, 200, 400, 500, 700, 900, 1100, 1200]

    ragas = ['Yaman', 'Bhairavi', 'Bilaval']

    for raga in ragas:
        print(f"\n--- {raga} Raga ---")

        hmm = ShrutiHMM(raga)
        fst = ShrutiFST(raga)

        hmm_results = hmm.correct_sequence(westernized)
        fst_results = fst.correct_sequence(westernized)

        print("HMM corrections:")
        for i, result in enumerate(hmm_results):
            print(f"  {westernized[i]:4.0f}c -> {result.label:4s} ({result.pitch:4.0f}c)")

        print("FST corrections:")
        for i, result in enumerate(fst_results):
            print(f"  {westernized[i]:4.0f}c -> {result.label:4s} ({result.pitch:4.0f}c)")


def performance_benchmark():
    """Benchmark performance of both models"""

    print("\n=== Performance Benchmark ===")

    import time

    # Create longer test sequence
    long_sequence = [i * 100 % 1200 for i in range(100)]  # 100 notes

    hmm = ShrutiHMM('Yaman')
    fst = ShrutiFST('Yaman')

    # Benchmark HMM
    start_time = time.time()
    hmm_results = hmm.correct_sequence(long_sequence)
    hmm_time = time.time() - start_time

    # Benchmark FST
    start_time = time.time()
    fst_results = fst.correct_sequence(long_sequence)
    fst_time = time.time() - start_time

    print(f"Sequence length: {len(long_sequence)} notes")
    print(f"HMM processing time: {hmm_time:.4f} seconds")
    print(f"FST processing time: {fst_time:.4f} seconds")
    print(f"Speed ratio (FST/HMM): {fst_time/hmm_time:.2f}x")


def main():
    """Main function to run all tests and demonstrations"""

    print("ShrutiSense: Symbolic Music Correction Tool for Cultural Restoration")
    print("=" * 70)

    # Run main comparative analysis
    results = run_comparative_analysis()

    # Additional analyses
    analyze_transition_matrices()
    test_melodic_direction_detection()
    test_fst_correction_rules()
    demonstrate_raga_specificity()
    performance_benchmark()

    # Export results
    export_results_to_json(results)

    print("\n" + "=" * 70)
    print("Analysis complete! Check the exported JSON file for detailed results.")
    print("This implementation provides a foundation for microtonal music correction")
    print("that can be extended with additional ragas, more sophisticated grammar")
    print("rules, and integration with audio processing pipelines.")


if __name__ == "__main__":
    main()