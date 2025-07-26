"""
ShrutiSense: Comprehensive Evaluation Suite for Symbolic Music Correction
Task 1: Correcting Incorrect Sequences

This program evaluates HMM and FST models on their ability to correct
sequences that have been artificially corrupted from ground truth.
"""

import numpy as np
from scipy.stats import norm
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import networkx as nx
from collections import defaultdict
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from itertools import combinations
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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

class RagaGrammar:
    """Encodes raga-specific grammar constraints"""

    def __init__(self, raga_name: str):
        self.raga_name = raga_name
        self.arohana = []
        self.avarohana = []
        self.vadi = None
        self.samvadi = None
        self.varjya = []
        self.pakad = []
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
            },
            'Kalyan': {
                'arohana': ['Sa', 'Re2', 'Ga2', 'Ma2', 'Pa', 'Dha2', 'Ni2', 'Sa_'],
                'avarohana': ['Sa_', 'Ni2', 'Dha2', 'Pa', 'Ma2', 'Ga2', 'Re2', 'Sa'],
                'vadi': 'Ga2',
                'samvadi': 'Ni2',
                'varjya': ['Ma1'],
                'pakad': [['Sa', 'Re2', 'Ga2', 'Ma2'], ['Pa', 'Ma2', 'Ga2']]
            },
            'Khamaaj': {
                'arohana': ['Sa', 'Re2', 'Ga2', 'Ma1', 'Pa', 'Dha2', 'Ni1', 'Sa_'],
                'avarohana': ['Sa_', 'Ni1', 'Dha2', 'Pa', 'Ma1', 'Ga2', 'Re2', 'Sa'],
                'vadi': 'Pa',
                'samvadi': 'Sa',
                'varjya': ['Ni2'],
                'pakad': [['Sa', 'Ga2', 'Ma1', 'Pa'], ['Pa', 'Dha2', 'Ni1']]
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
        """Get allowed transitions based on raga grammar"""
        scale = self.arohana if ascending else self.avarohana
        transitions = defaultdict(list)

        for i in range(len(scale) - 1):
            current = scale[i]
            next_note = scale[i + 1]
            transitions[current].append(next_note)
            transitions[current].append(current)  # Stay on same note

            if i < len(scale) - 2:
                transitions[current].append(scale[i + 2])  # Skip movement

        return dict(transitions)

    def get_transition_probability(self, from_shruti: str, to_shruti: str,
                                 ascending: bool = True) -> float:
        """Get transition probability based on raga grammar"""
        allowed = self.get_allowed_transitions(ascending)
        if to_shruti not in allowed.get(from_shruti, []):
            return 0.0

        scale = self.arohana if ascending else self.avarohana
        try:
            from_idx = scale.index(from_shruti)
            to_idx = scale.index(to_shruti)

            if from_idx == to_idx:
                return 0.3  # Same note
            elif abs(from_idx - to_idx) == 1:
                return 0.6  # Scale-wise movement
            elif abs(from_idx - to_idx) == 2:
                return 0.1  # Skip movement
            return 0.05
        except ValueError:
            return 0.1

class ShrutiModel(ABC):
    """Abstract base class for Shruti correction models"""

    def __init__(self, raga: str):
        self.raga = raga
        self.grammar = RagaGrammar(raga)
        self.shrutis = ShrutiUtils.get_shruti_list()

    @abstractmethod
    def correct_sequence(self, westernized_sequence: List[float]) -> List[CorrectionResult]:
        """Correct a sequence of westernized pitches"""
        pass

class ShrutiHMM(ShrutiModel):
    """Grammar-Constrained Shruti Hidden Markov Model"""

    def __init__(self, raga: str, sigma: float = 25.0):
        super().__init__(raga)
        self.sigma = sigma
        self.active_shrutis = self._get_active_shrutis()
        self.n_states = len(self.active_shrutis)
        self.state_to_idx = {s: i for i, s in enumerate(self.active_shrutis)}
        self.idx_to_state = {i: s for i, s in enumerate(self.active_shrutis)}

        self.pi = self._initialize_start_probabilities()
        self.A_up = self._initialize_transition_matrix(ascending=True)
        self.A_down = self._initialize_transition_matrix(ascending=False)
        self.B = self._initialize_emission_matrix()

    def _get_active_shrutis(self) -> List[str]:
        """Get active Shrutis for the raga"""
        active = set(self.grammar.arohana + self.grammar.avarohana)
        return sorted(active, key=lambda x: ShrutiUtils.shruti_to_cents(x))

    def _initialize_start_probabilities(self) -> np.ndarray:
        """Initialize start state probabilities"""
        pi = np.zeros(self.n_states)
        if 'Sa' in self.state_to_idx:
            pi[self.state_to_idx['Sa']] = 0.5

        remaining_prob = 1.0 - pi.sum()
        for i in range(self.n_states):
            if pi[i] == 0:
                pi[i] = remaining_prob / (self.n_states - 1)
        return pi

    def _initialize_transition_matrix(self, ascending: bool = True) -> np.ndarray:
        """Initialize transition matrix based on raga grammar"""
        A = np.zeros((self.n_states, self.n_states))

        for i, from_shruti in enumerate(self.active_shrutis):
            for j, to_shruti in enumerate(self.active_shrutis):
                prob = self.grammar.get_transition_probability(
                    from_shruti, to_shruti, ascending
                )
                A[i, j] = prob

        for i in range(self.n_states):
            row_sum = A[i, :].sum()
            if row_sum > 0:
                A[i, :] /= row_sum
            else:
                A[i, :] = 1.0 / self.n_states
        return A

    def _initialize_emission_matrix(self) -> np.ndarray:
        """Initialize emission matrix for Gaussian emissions"""
        emissions = np.zeros((self.n_states, 2))
        for i, shruti in enumerate(self.active_shrutis):
            emissions[i, 0] = ShrutiUtils.shruti_to_cents(shruti)
            emissions[i, 1] = self.sigma
        return emissions

    def _compute_emission_probability(self, state_idx: int, observation: float) -> float:
        """Compute emission probability for Gaussian model"""
        mean = self.B[state_idx, 0]
        std = self.B[state_idx, 1]
        return norm.pdf(observation, mean, std)

    def _detect_melodic_direction(self, observations: List[float]) -> List[bool]:
        """Detect melodic direction for each transition"""
        directions = []
        for i in range(len(observations) - 1):
            ascending = observations[i + 1] > observations[i]
            directions.append(ascending)
        return directions

    def viterbi_decode(self, observations: List[float]) -> Tuple[List[int], float]:
        """Viterbi algorithm for finding most likely state sequence"""
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        directions = self._detect_melodic_direction(observations)

        # Initialization
        for i in range(self.n_states):
            emission_prob = self._compute_emission_probability(i, observations[0])
            delta[0, i] = np.log(self.pi[i]) + np.log(emission_prob + 1e-10)
            psi[0, i] = 0

        # Recursion
        for t in range(1, T):
            ascending = directions[t-1] if t-1 < len(directions) else True
            A = self.A_up if ascending else self.A_down

            for j in range(self.n_states):
                emission_prob = self._compute_emission_probability(j, observations[t])
                scores = delta[t-1, :] + np.log(A[:, j] + 1e-10)
                best_prev = np.argmax(scores)
                delta[t, j] = scores[best_prev] + np.log(emission_prob + 1e-10)
                psi[t, j] = best_prev

        # Termination
        best_final = np.argmax(delta[T-1, :])
        best_prob = delta[T-1, best_final]

        # Backtrack
        path = [0] * T
        path[T-1] = best_final
        for t in range(T-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]

        return path, best_prob

    def correct_sequence(self, westernized_sequence: List[float]) -> List[CorrectionResult]:
        """Correct a sequence using HMM"""
        if not westernized_sequence:
            return []

        state_path, log_prob = self.viterbi_decode(westernized_sequence)
        results = []

        for i, (state_idx, original_pitch) in enumerate(zip(state_path, westernized_sequence)):
            shruti_label = self.idx_to_state[state_idx]
            corrected_pitch = ShrutiUtils.shruti_to_cents(shruti_label)
            error = corrected_pitch - original_pitch
            confidence = self._compute_emission_probability(state_idx, original_pitch)

            results.append(CorrectionResult(
                label=shruti_label,
                pitch=corrected_pitch,
                error=error,
                confidence=confidence
            ))

        return results

class ShrutiFST(ShrutiModel):
    """Shruti-based Finite-State Transducer"""

    def __init__(self, raga: str, tolerance: float = 50.0):
        super().__init__(raga)
        self.tolerance = tolerance
        self.fst_graph = self._build_fst_graph()
        self.current_state = 'Sa'

    def _build_fst_graph(self) -> nx.DiGraph:
        """Build FST graph with states and transitions"""
        G = nx.DiGraph()
        active_shrutis = set(self.grammar.arohana + self.grammar.avarohana)

        for shruti in active_shrutis:
            G.add_node(shruti, cents=ShrutiUtils.shruti_to_cents(shruti))

        for from_shruti in active_shrutis:
            allowed_up = self.grammar.get_allowed_transitions(ascending=True)
            for to_shruti in allowed_up.get(from_shruti, []):
                if to_shruti in active_shrutis:
                    weight = self.grammar.get_transition_probability(
                        from_shruti, to_shruti, ascending=True
                    )
                    G.add_edge(from_shruti, to_shruti, weight=weight, direction='up')

            allowed_down = self.grammar.get_allowed_transitions(ascending=False)
            for to_shruti in allowed_down.get(from_shruti, []):
                if to_shruti in active_shrutis:
                    weight = self.grammar.get_transition_probability(
                        from_shruti, to_shruti, ascending=False
                    )
                    G.add_edge(from_shruti, to_shruti, weight=weight, direction='down')

        return G

    def _find_nearest_shruti(self, cents: float, current_state: str = None) -> Tuple[str, float]:
        """Find nearest valid Shruti with FST constraints"""
        candidates = []

        if current_state and current_state in self.fst_graph:
            neighbors = list(self.fst_graph.neighbors(current_state))
            for neighbor in neighbors:
                neighbor_cents = self.fst_graph.nodes[neighbor]['cents']
                distance = abs(cents - neighbor_cents)
                if distance <= self.tolerance:
                    edge_weight = self.fst_graph[current_state][neighbor]['weight']
                    score = edge_weight * (1.0 - distance / self.tolerance)
                    candidates.append((neighbor, score))

        if not candidates:
            for shruti in self.fst_graph.nodes():
                shruti_cents = self.fst_graph.nodes[shruti]['cents']
                distance = abs(cents - shruti_cents)
                if distance <= self.tolerance:
                    score = 1.0 - distance / self.tolerance
                    candidates.append((shruti, score))

        if not candidates:
            best_shruti = min(
                self.fst_graph.nodes(),
                key=lambda s: abs(cents - self.fst_graph.nodes[s]['cents'])
            )
            return best_shruti, 0.1

        best_shruti, best_score = max(candidates, key=lambda x: x[1])
        return best_shruti, best_score

    def correct_sequence(self, westernized_sequence: List[float]) -> List[CorrectionResult]:
        """Correct sequence using FST"""
        if not westernized_sequence:
            return []

        results = []
        current_state = 'Sa'

        for cents in westernized_sequence:
            shruti, confidence = self._find_nearest_shruti(cents, current_state)
            corrected_pitch = ShrutiUtils.shruti_to_cents(shruti)
            error = corrected_pitch - cents

            results.append(CorrectionResult(
                label=shruti,
                pitch=corrected_pitch,
                error=error,
                confidence=confidence
            ))
            current_state = shruti

        return results

class SequenceGenerator:
    """Generate test sequences for evaluation"""

    def __init__(self, raga: str):
        self.raga = raga
        self.grammar = RagaGrammar(raga)

    def generate_melodic_sequence(self, length: int = 50, pattern_type: str = 'mixed') -> List[float]:
        """Generate a melodic sequence in cents"""
        if pattern_type == 'ascending':
            base_notes = self.grammar.arohana * (length // len(self.grammar.arohana) + 1)
        elif pattern_type == 'descending':
            base_notes = self.grammar.avarohana * (length // len(self.grammar.avarohana) + 1)
        else:  # mixed
            base_notes = []
            current_scale = self.grammar.arohana
            ascending = True

            for i in range(length):
                if i % 8 == 0:  # Change direction every 8 notes
                    ascending = not ascending
                    current_scale = self.grammar.arohana if ascending else self.grammar.avarohana

                note_idx = i % len(current_scale)
                base_notes.append(current_scale[note_idx])

        # Convert to cents
        sequence = [ShrutiUtils.shruti_to_cents(note) for note in base_notes[:length]]

        # Add some musical variation (ornaments)
        for i in range(len(sequence)):
            if random.random() < 0.1:  # 10% chance of ornament
                sequence[i] += random.uniform(-20, 20)  # Small pitch variation

        return sequence

    def corrupt_sequence(self, correct_sequence: List[float], corruption_level: float = 0.3) -> List[float]:
        """Corrupt a sequence to simulate westernization errors"""
        corrupted = correct_sequence.copy()
        n_corruptions = int(len(corrupted) * corruption_level)

        corruption_indices = random.sample(range(len(corrupted)), n_corruptions)

        for idx in corruption_indices:
            corruption_type = random.choice(['pitch_shift', 'quantization', 'random'])

            if corruption_type == 'pitch_shift':
                # Shift to nearest 12-TET note
                cents = corrupted[idx]
                tet_note = round(cents / 100) * 100
                corrupted[idx] = tet_note
            elif corruption_type == 'quantization':
                # Add random quantization error
                corrupted[idx] += random.uniform(-50, 50)
            else:  # random
                # Replace with random valid note
                valid_notes = [ShrutiUtils.shruti_to_cents(s) for s in self.grammar.arohana]
                corrupted[idx] = random.choice(valid_notes)

        return corrupted

class ComprehensiveEvaluator:
    """Comprehensive evaluation suite for correction models"""

    def __init__(self, ragas: List[str] = None):
        self.ragas = ragas or ['Yaman', 'Bhairavi', 'Bilaval', 'Kalyan', 'Khamaaj']
        self.results = []

    def evaluate_models(self, n_simulations: int = 100, sequence_lengths: List[int] = None,
                       corruption_levels: List[float] = None):
        """Run comprehensive evaluation"""
        sequence_lengths = sequence_lengths or [30, 50, 100, 150]
        corruption_levels = corruption_levels or [0.1, 0.2, 0.3, 0.4, 0.5]

        print("Starting comprehensive evaluation...")
        print(f"Ragas: {self.ragas}")
        print(f"Sequence lengths: {sequence_lengths}")
        print(f"Corruption levels: {corruption_levels}")
        print(f"Simulations per configuration: {n_simulations}")

        total_configs = len(self.ragas) * len(sequence_lengths) * len(corruption_levels)
        config_count = 0

        for raga in self.ragas:
            for seq_length in sequence_lengths:
                for corruption_level in corruption_levels:
                    config_count += 1
                    print(f"Configuration {config_count}/{total_configs}: {raga}, length={seq_length}, corruption={corruption_level}")

                    config_results = self._evaluate_configuration(
                        raga, seq_length, corruption_level, n_simulations
                    )
                    self.results.extend(config_results)

        print("Evaluation complete!")
        return self.results

    def _evaluate_configuration(self, raga: str, seq_length: int, corruption_level: float,
                               n_simulations: int) -> List[Dict]:
        """Evaluate a specific configuration"""
        generator = SequenceGenerator(raga)
        hmm_model = ShrutiHMM(raga)
        fst_model = ShrutiFST(raga)

        config_results = []

        for sim in range(n_simulations):
            # Generate ground truth and corrupted sequences
            pattern_type = random.choice(['ascending', 'descending', 'mixed'])
            ground_truth = generator.generate_melodic_sequence(seq_length, pattern_type)
            corrupted = generator.corrupt_sequence(ground_truth, corruption_level)

            # Time HMM correction
            start_time = time.time()
            hmm_results = hmm_model.correct_sequence(corrupted)
            hmm_time = time.time() - start_time

            # Time FST correction
            start_time = time.time()
            fst_results = fst_model.correct_sequence(corrupted)
            fst_time = time.time() - start_time

            # Calculate metrics
            hmm_metrics = self._calculate_metrics(hmm_results, ground_truth)
            fst_metrics = self._calculate_metrics(fst_results, ground_truth)

            # Store results
            result = {
                'raga': raga,
                'sequence_length': seq_length,
                'corruption_level': corruption_level,
                'pattern_type': pattern_type,
                'simulation': sim,
                'ground_truth': ground_truth,
                'corrupted': corrupted,
                'hmm_corrected': [r.pitch for r in hmm_results],
                'fst_corrected': [r.pitch for r in fst_results],
                'hmm_labels': [r.label for r in hmm_results],
                'fst_labels': [r.label for r in fst_results],
                'hmm_time': hmm_time,
                'fst_time': fst_time,
                'hmm_accuracy': hmm_metrics['accuracy'],
                'fst_accuracy': fst_metrics['accuracy'],
                'hmm_mean_error': hmm_metrics['mean_error'],
                'fst_mean_error': fst_metrics['mean_error'],
                'hmm_std_error': hmm_metrics['std_error'],
                'fst_std_error': fst_metrics['std_error']
            }
            config_results.append(result)

        return config_results

    def _calculate_metrics(self, results: List[CorrectionResult], ground_truth: List[float]) -> Dict:
        """Calculate evaluation metrics"""
        if len(results) != len(ground_truth):
            return {'accuracy': 0.0, 'mean_error': float('inf'), 'std_error': float('inf')}

        errors = [abs(r.pitch - gt) for r, gt in zip(results, ground_truth)]
        accurate_predictions = sum(1 for e in errors if e <= 25)  # Within 25 cents

        return {
            'accuracy': accurate_predictions / len(results),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors)
        }

class ResultsVisualizer:
    """Create comprehensive visualizations of evaluation results"""

    def __init__(self, results: List[Dict]):
        self.results = results
        self.df = pd.DataFrame(results)

    def create_all_plots(self):
        """Create all visualization plots"""
        print("Creating visualization plots...")

        # Set up the plot style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 24))

        # 1. Accuracy comparison
        plt.subplot(4, 3, 1)
        self._plot_accuracy_comparison()

        # 2. Mean error comparison
        plt.subplot(4, 3, 2)
        self._plot_error_comparison()

        # 3. Speed comparison
        plt.subplot(4, 3, 3)
        self._plot_speed_comparison()

        # 4. Accuracy vs corruption level
        plt.subplot(4, 3, 4)
        self._plot_accuracy_vs_corruption()

        # 5. Error vs sequence length
        plt.subplot(4, 3, 5)
        self._plot_error_vs_length()

        # 6. Raga-specific performance
        plt.subplot(4, 3, 6)
        self._plot_raga_performance()

        # 7. Pattern type analysis
        plt.subplot(4, 3, 7)
        self._plot_pattern_analysis()

        # 8. Error distribution
        plt.subplot(4, 3, 8)
        self._plot_error_distribution()

        # 9. Correlation matrix
        plt.subplot(4, 3, 9)
        self._plot_correlation_matrix()

        # 10. Statistical significance
        plt.subplot(4, 3, 10)
        self._plot_statistical_significance()

        # 11. Performance heatmap
        plt.subplot(4, 3, 11)
        self._plot_performance_heatmap()

        # 12. Detailed comparison
        plt.subplot(4, 3, 12)
        self._plot_detailed_comparison()

        plt.tight_layout()
        plt.savefig('shrutisense_correction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_accuracy_comparison(self):
        """Plot accuracy comparison between HMM and FST"""
        hmm_acc = self.df.groupby(['raga', 'corruption_level'])['hmm_accuracy'].mean()
        fst_acc = self.df.groupby(['raga', 'corruption_level'])['fst_accuracy'].mean()

        x = np.arange(len(hmm_acc))
        width = 0.35

        plt.bar(x - width/2, hmm_acc.values, width, label='HMM', alpha=0.8)
        plt.bar(x + width/2, fst_acc.values, width, label='FST', alpha=0.8)

        plt.xlabel('Configuration')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Comparison: HMM vs FST')
        plt.legend()
        plt.xticks(x, [f"{r}-{c:.1f}" for r, c in hmm_acc.index], rotation=45, ha='right')

    def _plot_error_comparison(self):
        """Plot mean error comparison"""
        hmm_err = self.df.groupby(['raga', 'corruption_level'])['hmm_mean_error'].mean()
        fst_err = self.df.groupby(['raga', 'corruption_level'])['fst_mean_error'].mean()

        x = np.arange(len(hmm_err))
        width = 0.35

        plt.bar(x - width/2, hmm_err.values, width, label='HMM', alpha=0.8)
        plt.bar(x + width/2, fst_err.values, width, label='FST', alpha=0.8)

        plt.xlabel('Configuration')
        plt.ylabel('Mean Error (cents)')
        plt.title('Mean Error Comparison: HMM vs FST')
        plt.legend()
        plt.xticks(x, [f"{r}-{c:.1f}" for r, c in hmm_err.index], rotation=45, ha='right')

    def _plot_speed_comparison(self):
        """Plot processing speed comparison"""
        hmm_speed = self.df.groupby('sequence_length')['hmm_time'].mean()
        fst_speed = self.df.groupby('sequence_length')['fst_time'].mean()

        plt.plot(hmm_speed.index, hmm_speed.values, 'o-', label='HMM', linewidth=2, markersize=8)
        plt.plot(fst_speed.index, fst_speed.values, 's-', label='FST', linewidth=2, markersize=8)

        plt.xlabel('Sequence Length')
        plt.ylabel('Processing Time (seconds)')
        plt.title('Processing Speed: HMM vs FST')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def _plot_accuracy_vs_corruption(self):
        """Plot accuracy vs corruption level"""
        corruption_levels = sorted(self.df['corruption_level'].unique())

        hmm_acc_mean = [self.df[self.df['corruption_level'] == cl]['hmm_accuracy'].mean()
                       for cl in corruption_levels]
        fst_acc_mean = [self.df[self.df['corruption_level'] == cl]['fst_accuracy'].mean()
                       for cl in corruption_levels]

        plt.plot(corruption_levels, hmm_acc_mean, 'o-', label='HMM', linewidth=2, markersize=8)
        plt.plot(corruption_levels, fst_acc_mean, 's-', label='FST', linewidth=2, markersize=8)

        plt.xlabel('Corruption Level')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Corruption Level')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def _plot_error_vs_length(self):
        """Plot error vs sequence length"""
        lengths = sorted(self.df['sequence_length'].unique())

        hmm_err_mean = [self.df[self.df['sequence_length'] == l]['hmm_mean_error'].mean()
                       for l in lengths]
        fst_err_mean = [self.df[self.df['sequence_length'] == l]['fst_mean_error'].mean()
                       for l in lengths]

        plt.plot(lengths, hmm_err_mean, 'o-', label='HMM', linewidth=2, markersize=8)
        plt.plot(lengths, fst_err_mean, 's-', label='FST', linewidth=2, markersize=8)

        plt.xlabel('Sequence Length')
        plt.ylabel('Mean Error (cents)')
        plt.title('Mean Error vs Sequence Length')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def _plot_raga_performance(self):
        """Plot performance by raga"""
        raga_data = []
        for raga in self.df['raga'].unique():
            raga_df = self.df[self.df['raga'] == raga]
            raga_data.append({
                'Raga': raga,
                'HMM_Accuracy': raga_df['hmm_accuracy'].mean(),
                'FST_Accuracy': raga_df['fst_accuracy'].mean(),
                'HMM_Error': raga_df['hmm_mean_error'].mean(),
                'FST_Error': raga_df['fst_mean_error'].mean()
            })

        raga_df = pd.DataFrame(raga_data)
        x = np.arange(len(raga_df))
        width = 0.35

        plt.bar(x - width/2, raga_df['HMM_Accuracy'], width, label='HMM', alpha=0.8)
        plt.bar(x + width/2, raga_df['FST_Accuracy'], width, label='FST', alpha=0.8)

        plt.xlabel('Raga')
        plt.ylabel('Accuracy')
        plt.title('Performance by Raga')
        plt.legend()
        plt.xticks(x, raga_df['Raga'], rotation=45)

    def _plot_pattern_analysis(self):
        """Plot performance by pattern type"""
        pattern_data = []
        for pattern in self.df['pattern_type'].unique():
            pattern_df = self.df[self.df['pattern_type'] == pattern]
            pattern_data.append({
                'Pattern': pattern,
                'HMM_Accuracy': pattern_df['hmm_accuracy'].mean(),
                'FST_Accuracy': pattern_df['fst_accuracy'].mean()
            })

        pattern_df = pd.DataFrame(pattern_data)
        x = np.arange(len(pattern_df))
        width = 0.35

        plt.bar(x - width/2, pattern_df['HMM_Accuracy'], width, label='HMM', alpha=0.8)
        plt.bar(x + width/2, pattern_df['FST_Accuracy'], width, label='FST', alpha=0.8)

        plt.xlabel('Pattern Type')
        plt.ylabel('Accuracy')
        plt.title('Performance by Pattern Type')
        plt.legend()
        plt.xticks(x, pattern_df['Pattern'])

    def _plot_error_distribution(self):
        """Plot error distribution"""
        hmm_errors = self.df['hmm_mean_error'].values
        fst_errors = self.df['fst_mean_error'].values

        plt.hist(hmm_errors, bins=30, alpha=0.7, label='HMM', density=True)
        plt.hist(fst_errors, bins=30, alpha=0.7, label='FST', density=True)

        plt.xlabel('Mean Error (cents)')
        plt.ylabel('Density')
        plt.title('Error Distribution')
        plt.legend()

    def _plot_correlation_matrix(self):
        """Plot correlation matrix of key metrics"""
        correlation_cols = ['hmm_accuracy', 'fst_accuracy', 'hmm_mean_error', 'fst_mean_error',
                           'hmm_time', 'fst_time', 'corruption_level', 'sequence_length']

        corr_matrix = self.df[correlation_cols].corr()

        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

    def _plot_statistical_significance(self):
        """Plot statistical significance tests"""
        # Perform paired t-tests
        t_stat_acc, p_val_acc = stats.ttest_rel(self.df['hmm_accuracy'], self.df['fst_accuracy'])
        t_stat_err, p_val_err = stats.ttest_rel(self.df['hmm_mean_error'], self.df['fst_mean_error'])
        t_stat_time, p_val_time = stats.ttest_rel(self.df['hmm_time'], self.df['fst_time'])

        metrics = ['Accuracy', 'Mean Error', 'Processing Time']
        p_values = [p_val_acc, p_val_err, p_val_time]
        significance = ['Significant' if p < 0.05 else 'Not Significant' for p in p_values]

        colors = ['green' if s == 'Significant' else 'red' for s in significance]

        plt.bar(metrics, [-np.log10(p) for p in p_values], color=colors, alpha=0.7)
        plt.axhline(y=-np.log10(0.05), color='black', linestyle='--', label='p=0.05')
        plt.xlabel('Metric')
        plt.ylabel('-log10(p-value)')
        plt.title('Statistical Significance Tests\n(HMM vs FST)')
        plt.legend()
        plt.xticks(rotation=45)

    def _plot_performance_heatmap(self):
        """Plot performance heatmap"""
        # Create pivot table for heatmap
        pivot_data = self.df.groupby(['raga', 'corruption_level'])['hmm_accuracy'].mean().unstack()

        sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', center=0.5,
                   square=False, linewidths=0.5, fmt='.3f')
        plt.title('HMM Accuracy Heatmap\n(Raga vs Corruption Level)')
        plt.xlabel('Corruption Level')
        plt.ylabel('Raga')

    def _plot_detailed_comparison(self):
        """Plot detailed comparison"""
        # Box plot comparing HMM and FST accuracy
        data_to_plot = [self.df['hmm_accuracy'].values, self.df['fst_accuracy'].values]

        box_plot = plt.boxplot(data_to_plot, labels=['HMM', 'FST'], patch_artist=True)

        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)

        plt.ylabel('Accuracy')
        plt.title('Detailed Accuracy Comparison')
        plt.grid(True, alpha=0.3)

def print_example_sequences(results: List[Dict], n_examples: int = 3):
    """Print example sequences for analysis"""
    print("\n" + "="*80)
    print("EXAMPLE SEQUENCES")
    print("="*80)

    # Select diverse examples
    selected_results = []

    # Get examples with different characteristics
    for raga in ['Yaman', 'Bhairavi']:
        for corruption in [0.2, 0.4]:
            matching = [r for r in results if r['raga'] == raga and r['corruption_level'] == corruption]
            if matching:
                selected_results.append(matching[0])
                if len(selected_results) >= n_examples:
                    break
        if len(selected_results) >= n_examples:
            break

    for i, result in enumerate(selected_results[:n_examples], 1):
        print(f"\nExample {i}: {result['raga']} raga, corruption={result['corruption_level']:.1f}")
        print(f"Pattern: {result['pattern_type']}, Length: {result['sequence_length']}")
        print("-" * 60)

        # Print first 15 notes for readability
        n_display = min(15, len(result['ground_truth']))

        print("Original (Ground Truth):")
        truth_cents = [f"{c:4.0f}" for c in result['ground_truth'][:n_display]]
        print("  Cents: " + " ".join(truth_cents))

        print("\nCorrupted Input:")
        corrupt_cents = [f"{c:4.0f}" for c in result['corrupted'][:n_display]]
        print("  Cents: " + " ".join(corrupt_cents))

        print("\nHMM Correction:")
        hmm_cents = [f"{c:4.0f}" for c in result['hmm_corrected'][:n_display]]
        hmm_labels = result['hmm_labels'][:n_display]
        print("  Cents: " + " ".join(hmm_cents))
        print("  Labels:" + " ".join(f"{l:>4}" for l in hmm_labels))

        print("\nFST Correction:")
        fst_cents = [f"{c:4.0f}" for c in result['fst_corrected'][:n_display]]
        fst_labels = result['fst_labels'][:n_display]
        print("  Cents: " + " ".join(fst_cents))
        print("  Labels:" + " ".join(f"{l:>4}" for l in fst_labels))

        print(f"\nPerformance:")
        print(f"  HMM: Accuracy={result['hmm_accuracy']:.3f}, Error={result['hmm_mean_error']:.1f}c, Time={result['hmm_time']:.4f}s")
        print(f"  FST: Accuracy={result['fst_accuracy']:.3f}, Error={result['fst_mean_error']:.1f}c, Time={result['fst_time']:.4f}s")

def print_summary_statistics(results: List[Dict]):
    """Print comprehensive summary statistics"""
    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"\nDataset Overview:")
    print(f"  Total simulations: {len(df)}")
    print(f"  Ragas: {list(df['raga'].unique())}")
    print(f"  Sequence lengths: {sorted(df['sequence_length'].unique())}")
    print(f"  Corruption levels: {sorted(df['corruption_level'].unique())}")

    print(f"\nOverall Performance:")
    print(f"  HMM - Mean Accuracy: {df['hmm_accuracy'].mean():.3f} ± {df['hmm_accuracy'].std():.3f}")
    print(f"  FST - Mean Accuracy: {df['fst_accuracy'].mean():.3f} ± {df['fst_accuracy'].std():.3f}")
    print(f"  HMM - Mean Error: {df['hmm_mean_error'].mean():.1f} ± {df['hmm_mean_error'].std():.1f} cents")
    print(f"  FST - Mean Error: {df['fst_mean_error'].mean():.1f} ± {df['fst_mean_error'].std():.1f} cents")
    print(f"  HMM - Mean Time: {df['hmm_time'].mean():.4f} ± {df['hmm_time'].std():.4f} seconds")
    print(f"  FST - Mean Time: {df['fst_time'].mean():.4f} ± {df['fst_time'].std():.4f} seconds")

    # Statistical significance
    t_stat_acc, p_val_acc = stats.ttest_rel(df['hmm_accuracy'], df['fst_accuracy'])
    t_stat_err, p_val_err = stats.ttest_rel(df['hmm_mean_error'], df['fst_mean_error'])

    print(f"\nStatistical Significance (paired t-test):")
    print(f"  Accuracy difference: t={t_stat_acc:.3f}, p={p_val_acc:.6f}")
    print(f"  Error difference: t={t_stat_err:.3f}, p={p_val_err:.6f}")

    if p_val_acc < 0.05:
        better_acc = "HMM" if df['hmm_accuracy'].mean() > df['fst_accuracy'].mean() else "FST"
        print(f"  → {better_acc} significantly better for accuracy")
    else:
        print(f"  → No significant difference in accuracy")

    if p_val_err < 0.05:
        better_err = "HMM" if df['hmm_mean_error'].mean() < df['fst_mean_error'].mean() else "FST"
        print(f"  → {better_err} significantly better for error")
    else:
        print(f"  → No significant difference in error")

    print(f"\nPerformance by Raga:")
    for raga in df['raga'].unique():
        raga_df = df[df['raga'] == raga]
        print(f"  {raga}:")
        print(f"    HMM: Acc={raga_df['hmm_accuracy'].mean():.3f}, Err={raga_df['hmm_mean_error'].mean():.1f}c")
        print(f"    FST: Acc={raga_df['fst_accuracy'].mean():.3f}, Err={raga_df['fst_mean_error'].mean():.1f}c")

def main():
    """Main function to run the correction evaluation"""
    print("ShrutiSense: Symbolic Music Correction Evaluation Suite")
    print("Task 1: Correcting Incorrect Sequences")
    print("="*80)

    # Initialize evaluator
    evaluator = ComprehensiveEvaluator()

    # Run evaluation (reduced numbers for demo - increase for full evaluation)
    print("Running evaluation...")
    results = evaluator.evaluate_models(
        n_simulations=20,  # Increase to 100+ for full evaluation
        sequence_lengths=[30, 50, 100],
        corruption_levels=[0.2, 0.3, 0.4]
    )

    # Create visualizations
    visualizer = ResultsVisualizer(results)
    visualizer.create_all_plots()

    # Print statistics and examples
    print_summary_statistics(results)
    print_example_sequences(results)

    # Save results
    with open('correction_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\nEvaluation complete! Results saved to 'correction_results.json'")
    print("Visualization saved as 'shrutisense_correction_analysis.png'")

if __name__ == "__main__":
    main()
