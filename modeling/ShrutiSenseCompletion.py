"""
ShrutiSense: Comprehensive Evaluation Suite for Symbolic Music Completion
Task 2: Filling Missing Values in Sequences

This program evaluates HMM and FST models on their ability to predict
missing notes in melodic sequences using contextual information.
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
import random
from itertools import combinations

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

@dataclass
class CompletionResult:
    """Data structure for completion results"""
    label: str
    pitch: float
    confidence: float = 0.0
    position: int = 0  # Position in sequence where prediction was made

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

class ShrutiCompletionModel(ABC):
    """Abstract base class for Shruti completion models"""

    def __init__(self, raga: str):
        self.raga = raga
        self.grammar = RagaGrammar(raga)
        self.shrutis = ShrutiUtils.get_shruti_list()

    @abstractmethod
    def complete_sequence(self, partial_sequence: List[Optional[float]],
                         missing_positions: List[int]) -> List[CompletionResult]:
        """Complete a sequence with missing values"""
        pass

class ShrutiCompletionHMM(ShrutiCompletionModel):
    """HMM-based completion model using forward-backward algorithm"""

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
        if observation is None:
            return 1.0  # Uniform probability for missing observations

        mean = self.B[state_idx, 0]
        std = self.B[state_idx, 1]
        return norm.pdf(observation, mean, std)

    def _detect_melodic_direction(self, observations: List[Optional[float]]) -> List[bool]:
        """Detect melodic direction, handling missing values"""
        directions = []

        for i in range(len(observations) - 1):
            current = observations[i]
            next_obs = observations[i + 1]

            # If either observation is missing, use previous direction or default to ascending
            if current is None or next_obs is None:
                if directions:
                    directions.append(directions[-1])  # Use previous direction
                else:
                    directions.append(True)  # Default to ascending
            else:
                directions.append(next_obs > current)

        return directions

    def _forward_backward(self, observations: List[Optional[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Forward-backward algorithm for missing data"""
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        beta = np.zeros((T, self.n_states))

        directions = self._detect_melodic_direction(observations)

        # Forward pass
        # Initialization
        for i in range(self.n_states):
            emission_prob = self._compute_emission_probability(i, observations[0])
            alpha[0, i] = self.pi[i] * emission_prob

        # Normalize
        alpha[0, :] /= (alpha[0, :].sum() + 1e-10)

        # Recursion
        for t in range(1, T):
            ascending = directions[t-1] if t-1 < len(directions) else True
            A = self.A_up if ascending else self.A_down

            for j in range(self.n_states):
                emission_prob = self._compute_emission_probability(j, observations[t])
                alpha[t, j] = emission_prob * np.sum(alpha[t-1, :] * A[:, j])

            # Normalize to prevent underflow
            alpha[t, :] /= (alpha[t, :].sum() + 1e-10)

        # Backward pass
        # Initialization
        beta[T-1, :] = 1.0

        # Recursion
        for t in range(T-2, -1, -1):
            ascending = directions[t] if t < len(directions) else True
            A = self.A_up if ascending else self.A_down

            for i in range(self.n_states):
                beta[t, i] = 0
                for j in range(self.n_states):
                    emission_prob = self._compute_emission_probability(j, observations[t+1])
                    beta[t, i] += A[i, j] * emission_prob * beta[t+1, j]

            # Normalize
            beta[t, :] /= (beta[t, :].sum() + 1e-10)

        return alpha, beta

    def complete_sequence(self, partial_sequence: List[Optional[float]],
                         missing_positions: List[int]) -> List[CompletionResult]:
        """Complete sequence using forward-backward algorithm"""
        if not partial_sequence:
            return []

        # Run forward-backward algorithm
        alpha, beta = self._forward_backward(partial_sequence)

        # Compute posterior probabilities for missing positions
        results = []

        for pos in missing_positions:
            if pos >= len(partial_sequence):
                continue

            # Compute gamma (posterior state probabilities)
            gamma = alpha[pos, :] * beta[pos, :]
            gamma /= (gamma.sum() + 1e-10)

            # Find most likely state
            best_state_idx = np.argmax(gamma)
            confidence = gamma[best_state_idx]

            shruti_label = self.idx_to_state[best_state_idx]
            predicted_pitch = ShrutiUtils.shruti_to_cents(shruti_label)

            results.append(CompletionResult(
                label=shruti_label,
                pitch=predicted_pitch,
                confidence=confidence,
                position=pos
            ))

        return results

class ShrutiCompletionFST(ShrutiCompletionModel):
    """FST-based completion using contextual grammar rules"""

    def __init__(self, raga: str, context_window: int = 3):
        super().__init__(raga)
        self.context_window = context_window
        self.fst_graph = self._build_fst_graph()

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

    def _get_context(self, sequence: List[Optional[float]], position: int) -> Dict[str, List[str]]:
        """Extract context around missing position"""
        context = {'before': [], 'after': []}

        # Extract before context
        start_pos = max(0, position - self.context_window)
        for i in range(start_pos, position):
            if sequence[i] is not None:
                shruti = ShrutiUtils.cents_to_shruti(sequence[i])
                if shruti:
                    context['before'].append(shruti)

        # Extract after context
        end_pos = min(len(sequence), position + self.context_window + 1)
        for i in range(position + 1, end_pos):
            if sequence[i] is not None:
                shruti = ShrutiUtils.cents_to_shruti(sequence[i])
                if shruti:
                    context['after'].append(shruti)

        return context

    def _score_candidate(self, candidate: str, context: Dict[str, List[str]],
                        position: int, sequence_length: int) -> float:
        """Score a candidate Shruti based on context"""
        score = 0.0

        # Base score from raga membership
        if candidate in self.grammar.arohana or candidate in self.grammar.avarohana:
            score += 0.5

        # Score based on transitions from before context
        if context['before']:
            last_before = context['before'][-1]
            if last_before in self.fst_graph and candidate in self.fst_graph:
                if self.fst_graph.has_edge(last_before, candidate):
                    score += self.fst_graph[last_before][candidate]['weight']

        # Score based on transitions to after context
        if context['after']:
            first_after = context['after'][0]
            if candidate in self.fst_graph and first_after in self.fst_graph:
                if self.fst_graph.has_edge(candidate, first_after):
                    score += self.fst_graph[candidate][first_after]['weight']

        # Bonus for vadi/samvadi at strong positions
        if position % 4 == 0:  # Strong beat
            if candidate == self.grammar.vadi:
                score += 0.3
            elif candidate == self.grammar.samvadi:
                score += 0.2

        # Pattern matching bonus
        score += self._pattern_matching_score(candidate, context)

        return score

    def _pattern_matching_score(self, candidate: str, context: Dict[str, List[str]]) -> float:
        """Score based on pattern matching with pakad phrases"""
        score = 0.0

        for pakad_phrase in self.grammar.pakad:
            # Check if candidate completes a pakad phrase
            combined_context = context['before'] + [candidate] + context['after']

            for i in range(len(combined_context) - len(pakad_phrase) + 1):
                segment = combined_context[i:i + len(pakad_phrase)]
                if segment == pakad_phrase:
                    score += 0.4
                elif len(set(segment) & set(pakad_phrase)) >= len(pakad_phrase) * 0.7:
                    score += 0.2

        return score

    def complete_sequence(self, partial_sequence: List[Optional[float]],
                         missing_positions: List[int]) -> List[CompletionResult]:
        """Complete sequence using FST with contextual rules"""
        if not partial_sequence:
            return []

        results = []

        for pos in missing_positions:
            if pos >= len(partial_sequence):
                continue

            context = self._get_context(partial_sequence, pos)
            candidates = list(self.fst_graph.nodes())

            # Score all candidates
            candidate_scores = []
            for candidate in candidates:
                score = self._score_candidate(candidate, context, pos, len(partial_sequence))
                candidate_scores.append((candidate, score))

            # Select best candidate
            if candidate_scores:
                best_candidate, best_score = max(candidate_scores, key=lambda x: x[1])

                # Normalize confidence
                total_score = sum(score for _, score in candidate_scores)
                confidence = best_score / (total_score + 1e-10) if total_score > 0 else 0.1

                predicted_pitch = ShrutiUtils.shruti_to_cents(best_candidate)

                results.append(CompletionResult(
                    label=best_candidate,
                    pitch=predicted_pitch,
                    confidence=confidence,
                    position=pos
                ))

        return results

class SequenceGenerator:
    """Generate test sequences for completion evaluation"""

    def __init__(self, raga: str):
        self.raga = raga
        self.grammar = RagaGrammar(raga)

    def generate_complete_sequence(self, length: int = 50, pattern_type: str = 'mixed') -> List[float]:
        """Generate a complete melodic sequence"""
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

        # Convert to cents and add musical variation
        sequence = [ShrutiUtils.shruti_to_cents(note) for note in base_notes[:length]]

        # Add some ornaments
        for i in range(len(sequence)):
            if random.random() < 0.1:
                sequence[i] += random.uniform(-15, 15)

        return sequence

    def create_missing_pattern(self, complete_sequence: List[float],
                             missing_rate: float = 0.3,
                             pattern_type: str = 'random') -> Tuple[List[Optional[float]], List[int]]:
        """Create missing value pattern in sequence"""
        partial_sequence = complete_sequence.copy()
        n_missing = int(len(complete_sequence) * missing_rate)

        if pattern_type == 'random':
            missing_positions = random.sample(range(len(complete_sequence)), n_missing)
        elif pattern_type == 'clustered':
            # Create clusters of missing values
            cluster_size = 3
            n_clusters = n_missing // cluster_size
            missing_positions = []

            for _ in range(n_clusters):
                start_pos = random.randint(0, len(complete_sequence) - cluster_size)
                for i in range(cluster_size):
                    if start_pos + i < len(complete_sequence):
                        missing_positions.append(start_pos + i)

            # Add remaining missing positions randomly
            remaining = n_missing - len(missing_positions)
            available_positions = [i for i in range(len(complete_sequence))
                                 if i not in missing_positions]
            if remaining > 0 and available_positions:
                additional = random.sample(available_positions, min(remaining, len(available_positions)))
                missing_positions.extend(additional)

        elif pattern_type == 'periodic':
            # Remove every k-th element
            period = int(1 / missing_rate)
            missing_positions = [i for i in range(0, len(complete_sequence), period)][:n_missing]

        else:  # structured - remove important structural notes
            # Focus on removing notes at phrase boundaries and strong beats
            structural_positions = [i for i in range(len(complete_sequence))
                                  if i % 4 == 0 or i % 8 == 7]  # Downbeats and phrase ends
            missing_positions = random.sample(structural_positions,
                                            min(n_missing, len(structural_positions)))

            # Fill remaining with random positions
            if len(missing_positions) < n_missing:
                remaining_positions = [i for i in range(len(complete_sequence))
                                     if i not in missing_positions]
                additional = random.sample(remaining_positions,
                                         min(n_missing - len(missing_positions),
                                             len(remaining_positions)))
                missing_positions.extend(additional)

        # Apply missing pattern
        for pos in missing_positions:
            partial_sequence[pos] = None

        return partial_sequence, sorted(missing_positions)

class ComprehensiveCompletionEvaluator:
    """Comprehensive evaluation suite for completion models"""

    def __init__(self, ragas: List[str] = None):
        self.ragas = ragas or ['Yaman', 'Bhairavi', 'Bilaval', 'Kalyan', 'Khamaaj']
        self.results = []

    def evaluate_models(self, n_simulations: int = 100, sequence_lengths: List[int] = None,
                       missing_rates: List[float] = None, missing_patterns: List[str] = None):
        """Run comprehensive completion evaluation"""
        sequence_lengths = sequence_lengths or [30, 50, 100, 150]
        missing_rates = missing_rates or [0.1, 0.2, 0.3, 0.4, 0.5]
        missing_patterns = missing_patterns or ['random', 'clustered', 'periodic', 'structured']

        print("Starting comprehensive completion evaluation...")
        print(f"Ragas: {self.ragas}")
        print(f"Sequence lengths: {sequence_lengths}")
        print(f"Missing rates: {missing_rates}")
        print(f"Missing patterns: {missing_patterns}")
        print(f"Simulations per configuration: {n_simulations}")

        total_configs = len(self.ragas) * len(sequence_lengths) * len(missing_rates) * len(missing_patterns)
        config_count = 0

        for raga in self.ragas:
            for seq_length in sequence_lengths:
                for missing_rate in missing_rates:
                    for missing_pattern in missing_patterns:
                        config_count += 1
                        print(f"Configuration {config_count}/{total_configs}: {raga}, "
                              f"length={seq_length}, missing={missing_rate}, pattern={missing_pattern}")

                        config_results = self._evaluate_configuration(
                            raga, seq_length, missing_rate, missing_pattern, n_simulations
                        )
                        self.results.extend(config_results)

        print("Completion evaluation complete!")
        return self.results

    def _evaluate_configuration(self, raga: str, seq_length: int, missing_rate: float,
                               missing_pattern: str, n_simulations: int) -> List[Dict]:
        """Evaluate a specific configuration"""
        generator = SequenceGenerator(raga)
        hmm_model = ShrutiCompletionHMM(raga)
        fst_model = ShrutiCompletionFST(raga)

        config_results = []

        for sim in range(n_simulations):
            # Generate complete sequence and create missing pattern
            pattern_type = random.choice(['ascending', 'descending', 'mixed'])
            complete_sequence = generator.generate_complete_sequence(seq_length, pattern_type)
            partial_sequence, missing_positions = generator.create_missing_pattern(
                complete_sequence, missing_rate, missing_pattern
            )

            # Time HMM completion
            start_time = time.time()
            hmm_results = hmm_model.complete_sequence(partial_sequence, missing_positions)
            hmm_time = time.time() - start_time

            # Time FST completion
            start_time = time.time()
            fst_results = fst_model.complete_sequence(partial_sequence, missing_positions)
            fst_time = time.time() - start_time

            # Calculate metrics
            hmm_metrics = self._calculate_completion_metrics(hmm_results, complete_sequence, missing_positions)
            fst_metrics = self._calculate_completion_metrics(fst_results, complete_sequence, missing_positions)

            # Store results
            result = {
                'raga': raga,
                'sequence_length': seq_length,
                'missing_rate': missing_rate,
                'missing_pattern': missing_pattern,
                'pattern_type': pattern_type,
                'simulation': sim,
                'complete_sequence': complete_sequence,
                'partial_sequence': [x if x is not None else -999 for x in partial_sequence],  # Replace None for JSON
                'missing_positions': missing_positions,
                'hmm_predictions': [r.pitch for r in hmm_results],
                'fst_predictions': [r.pitch for r in fst_results],
                'hmm_labels': [r.label for r in hmm_results],
                'fst_labels': [r.label for r in fst_results],
                'hmm_confidences': [r.confidence for r in hmm_results],
                'fst_confidences': [r.confidence for r in fst_results],
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

    def _calculate_completion_metrics(self, results: List[CompletionResult],
                                    complete_sequence: List[float],
                                    missing_positions: List[int]) -> Dict:
        """Calculate completion evaluation metrics"""
        if len(results) != len(missing_positions):
            return {'accuracy': 0.0, 'mean_error': float('inf'), 'std_error': float('inf')}

        errors = []
        accurate_predictions = 0

        for result, pos in zip(results, missing_positions):
            if pos < len(complete_sequence):
                true_value = complete_sequence[pos]
                error = abs(result.pitch - true_value)
                errors.append(error)

                if error <= 25:  # Within 25 cents
                    accurate_predictions += 1

        if not errors:
            return {'accuracy': 0.0, 'mean_error': float('inf'), 'std_error': float('inf')}

        return {
            'accuracy': accurate_predictions / len(errors),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors)
        }

class CompletionResultsVisualizer:
    """Create comprehensive visualizations for completion results"""

    def __init__(self, results: List[Dict]):
        self.results = results
        self.df = pd.DataFrame(results)

    def create_all_plots(self):
        """Create all visualization plots"""
        print("Creating completion visualization plots...")

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

        # 4. Accuracy vs missing rate
        plt.subplot(4, 3, 4)
        self._plot_accuracy_vs_missing_rate()

        # 5. Error vs sequence length
        plt.subplot(4, 3, 5)
        self._plot_error_vs_length()

        # 6. Missing pattern analysis
        plt.subplot(4, 3, 6)
        self._plot_missing_pattern_analysis()

        # 7. Raga-specific performance
        plt.subplot(4, 3, 7)
        self._plot_raga_performance()

        # 8. Confidence analysis
        plt.subplot(4, 3, 8)
        self._plot_confidence_analysis()

        # 9. Error distribution
        plt.subplot(4, 3, 9)
        self._plot_error_distribution()

        # 10. Performance heatmap
        plt.subplot(4, 3, 10)
        self._plot_performance_heatmap()

        # 11. Statistical significance
        plt.subplot(4, 3, 11)
        self._plot_statistical_significance()

        # 12. Model comparison boxplot
        plt.subplot(4, 3, 12)
        self._plot_model_comparison_boxplot()

        plt.tight_layout()
        plt.savefig('shrutisense_completion_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _plot_accuracy_comparison(self):
        """Plot accuracy comparison between HMM and FST"""
        hmm_acc = self.df.groupby(['raga', 'missing_rate'])['hmm_accuracy'].mean()
        fst_acc = self.df.groupby(['raga', 'missing_rate'])['fst_accuracy'].mean()

        x = np.arange(len(hmm_acc))
        width = 0.35

        plt.bar(x - width/2, hmm_acc.values, width, label='HMM', alpha=0.8)
        plt.bar(x + width/2, fst_acc.values, width, label='FST', alpha=0.8)

        plt.xlabel('Configuration')
        plt.ylabel('Accuracy')
        plt.title('Completion Accuracy: HMM vs FST')
        plt.legend()
        plt.xticks(x, [f"{r}-{mr:.1f}" for r, mr in hmm_acc.index], rotation=45, ha='right')

    def _plot_error_comparison(self):
        """Plot mean error comparison"""
        hmm_err = self.df.groupby(['raga', 'missing_rate'])['hmm_mean_error'].mean()
        fst_err = self.df.groupby(['raga', 'missing_rate'])['fst_mean_error'].mean()

        x = np.arange(len(hmm_err))
        width = 0.35

        plt.bar(x - width/2, hmm_err.values, width, label='HMM', alpha=0.8)
        plt.bar(x + width/2, fst_err.values, width, label='FST', alpha=0.8)

        plt.xlabel('Configuration')
        plt.ylabel('Mean Error (cents)')
        plt.title('Completion Error: HMM vs FST')
        plt.legend()
        plt.xticks(x, [f"{r}-{mr:.1f}" for r, mr in hmm_err.index], rotation=45, ha='right')

    def _plot_speed_comparison(self):
        """Plot processing speed comparison"""
        hmm_speed = self.df.groupby('sequence_length')['hmm_time'].mean()
        fst_speed = self.df.groupby('sequence_length')['fst_time'].mean()

        plt.plot(hmm_speed.index, hmm_speed.values, 'o-', label='HMM', linewidth=2, markersize=8)
        plt.plot(fst_speed.index, fst_speed.values, 's-', label='FST', linewidth=2, markersize=8)

        plt.xlabel('Sequence Length')
        plt.ylabel('Processing Time (seconds)')
        plt.title('Completion Speed: HMM vs FST')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def _plot_accuracy_vs_missing_rate(self):
        """Plot accuracy vs missing rate"""
        missing_rates = sorted(self.df['missing_rate'].unique())

        hmm_acc_mean = [self.df[self.df['missing_rate'] == mr]['hmm_accuracy'].mean()
                       for mr in missing_rates]
        fst_acc_mean = [self.df[self.df['missing_rate'] == mr]['fst_accuracy'].mean()
                       for mr in missing_rates]

        plt.plot(missing_rates, hmm_acc_mean, 'o-', label='HMM', linewidth=2, markersize=8)
        plt.plot(missing_rates, fst_acc_mean, 's-', label='FST', linewidth=2, markersize=8)

        plt.xlabel('Missing Rate')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Missing Rate')
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
        plt.title('Completion Error vs Sequence Length')
        plt.legend()
        plt.grid(True, alpha=0.3)

    def _plot_missing_pattern_analysis(self):
        """Plot performance by missing pattern type"""
        pattern_data = []
        for pattern in self.df['missing_pattern'].unique():
            pattern_df = self.df[self.df['missing_pattern'] == pattern]
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

        plt.xlabel('Missing Pattern')
        plt.ylabel('Accuracy')
        plt.title('Performance by Missing Pattern')
        plt.legend()
        plt.xticks(x, pattern_df['Pattern'], rotation=45)

    def _plot_raga_performance(self):
        """Plot performance by raga"""
        raga_data = []
        for raga in self.df['raga'].unique():
            raga_df = self.df[self.df['raga'] == raga]
            raga_data.append({
                'Raga': raga,
                'HMM_Accuracy': raga_df['hmm_accuracy'].mean(),
                'FST_Accuracy': raga_df['fst_accuracy'].mean()
            })

        raga_df = pd.DataFrame(raga_data)
        x = np.arange(len(raga_df))
        width = 0.35

        plt.bar(x - width/2, raga_df['HMM_Accuracy'], width, label='HMM', alpha=0.8)
        plt.bar(x + width/2, raga_df['FST_Accuracy'], width, label='FST', alpha=0.8)

        plt.xlabel('Raga')
        plt.ylabel('Accuracy')
        plt.title('Completion Performance by Raga')
        plt.legend()
        plt.xticks(x, raga_df['Raga'], rotation=45)

    def _plot_confidence_analysis(self):
        """Plot confidence analysis"""
        # Only plot if confidence data exists
        if 'hmm_confidences' in self.df.columns and len(self.df['hmm_confidences'].iloc[0]) > 0:
            hmm_conf = [np.mean(conf) for conf in self.df['hmm_confidences'] if conf]
            fst_conf = [np.mean(conf) for conf in self.df['fst_confidences'] if conf]

            if hmm_conf and fst_conf:
                plt.hist(hmm_conf, bins=20, alpha=0.7, label='HMM', density=True)
                plt.hist(fst_conf, bins=20, alpha=0.7, label='FST', density=True)

                plt.xlabel('Mean Confidence')
                plt.ylabel('Density')
                plt.title('Confidence Distribution')
                plt.legend()
            else:
                plt.text(0.5, 0.5, 'No confidence data available',
                        transform=plt.gca().transAxes, ha='center', va='center')
        else:
            plt.text(0.5, 0.5, 'No confidence data available',
                    transform=plt.gca().transAxes, ha='center', va='center')

    def _plot_error_distribution(self):
        """Plot error distribution"""
        hmm_errors = self.df['hmm_mean_error'].values
        fst_errors = self.df['fst_mean_error'].values

        plt.hist(hmm_errors, bins=30, alpha=0.7, label='HMM', density=True)
        plt.hist(fst_errors, bins=30, alpha=0.7, label='FST', density=True)

        plt.xlabel('Mean Error (cents)')
        plt.ylabel('Density')
        plt.title('Completion Error Distribution')
        plt.legend()

    def _plot_performance_heatmap(self):
        """Plot performance heatmap"""
        pivot_data = self.df.groupby(['raga', 'missing_rate'])['hmm_accuracy'].mean().unstack()

        sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', center=0.5,
                   square=False, linewidths=0.5, fmt='.3f')
        plt.title('HMM Completion Accuracy\n(Raga vs Missing Rate)')
        plt.xlabel('Missing Rate')
        plt.ylabel('Raga')

    def _plot_statistical_significance(self):
        """Plot statistical significance tests"""
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
        plt.title('Statistical Significance Tests\n(HMM vs FST Completion)')
        plt.legend()
        plt.xticks(rotation=45)

    def _plot_model_comparison_boxplot(self):
        """Plot detailed model comparison"""
        data_to_plot = [self.df['hmm_accuracy'].values, self.df['fst_accuracy'].values]

        box_plot = plt.boxplot(data_to_plot, labels=['HMM', 'FST'], patch_artist=True)

        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)

        plt.ylabel('Accuracy')
        plt.title('Detailed Completion Accuracy Comparison')
        plt.grid(True, alpha=0.3)

def print_completion_examples(results: List[Dict], n_examples: int = 3):
    """Print example completion sequences for analysis"""
    print("\n" + "="*80)
    print("COMPLETION EXAMPLES")
    print("="*80)

    # Select diverse examples
    selected_results = []

    for raga in ['Yaman', 'Bhairavi']:
        for missing_rate in [0.2, 0.4]:
            for missing_pattern in ['random', 'clustered']:
                matching = [r for r in results if (r['raga'] == raga and
                                                 r['missing_rate'] == missing_rate and
                                                 r['missing_pattern'] == missing_pattern)]
                if matching:
                    selected_results.append(matching[0])
                    if len(selected_results) >= n_examples:
                        break
            if len(selected_results) >= n_examples:
                break
        if len(selected_results) >= n_examples:
            break

    for i, result in enumerate(selected_results[:n_examples], 1):
        print(f"\nExample {i}: {result['raga']} raga, missing_rate={result['missing_rate']:.1f}")
        print(f"Pattern: {result['pattern_type']}, Missing pattern: {result['missing_pattern']}")
        print(f"Length: {result['sequence_length']}, Missing positions: {result['missing_positions'][:10]}...")
        print("-" * 60)

        # Display first 20 notes for readability
        n_display = min(20, len(result['complete_sequence']))

        print("Complete Sequence (Ground Truth):")
        complete_cents = []
        for j in range(n_display):
            complete_cents.append(f"{result['complete_sequence'][j]:4.0f}")
        print("  Cents: " + " ".join(complete_cents))

        print("\nPartial Sequence (with missing values):")
        partial_cents = []
        for j in range(n_display):
            if j in result['missing_positions']:
                partial_cents.append("  ??")
            else:
                val = result['partial_sequence'][j]
                if val != -999:  # Our sentinel value for None
                    partial_cents.append(f"{val:4.0f}")
                else:
                    partial_cents.append("  ??")
        print("  Cents: " + " ".join(partial_cents))

        print("\nHMM Predictions:")
        hmm_display = ["    " for _ in range(n_display)]
        hmm_labels_display = ["    " for _ in range(n_display)]
        for j, pos in enumerate(result['missing_positions']):
            if pos < n_display and j < len(result['hmm_predictions']):
                hmm_display[pos] = f"{result['hmm_predictions'][j]:4.0f}"
                hmm_labels_display[pos] = f"{result['hmm_labels'][j]:>4}"
        print("  Cents: " + " ".join(hmm_display))
        print("  Labels:" + " ".join(hmm_labels_display))

        print("\nFST Predictions:")
        fst_display = ["    " for _ in range(n_display)]
        fst_labels_display = ["    " for _ in range(n_display)]
        for j, pos in enumerate(result['missing_positions']):
            if pos < n_display and j < len(result['fst_predictions']):
                fst_display[pos] = f"{result['fst_predictions'][j]:4.0f}"
                fst_labels_display[pos] = f"{result['fst_labels'][j]:>4}"
        print("  Cents: " + " ".join(fst_display))
        print("  Labels:" + " ".join(fst_labels_display))

        print(f"\nPerformance:")
        print(f"  HMM: Accuracy={result['hmm_accuracy']:.3f}, Error={result['hmm_mean_error']:.1f}c, Time={result['hmm_time']:.4f}s")
        print(f"  FST: Accuracy={result['fst_accuracy']:.3f}, Error={result['fst_mean_error']:.1f}c, Time={result['fst_time']:.4f}s")

def print_completion_summary_statistics(results: List[Dict]):
    """Print comprehensive summary statistics for completion task"""
    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("COMPLETION SUMMARY STATISTICS")
    print("="*80)

    print(f"\nDataset Overview:")
    print(f"  Total simulations: {len(df)}")
    print(f"  Ragas: {list(df['raga'].unique())}")
    print(f"  Sequence lengths: {sorted(df['sequence_length'].unique())}")
    print(f"  Missing rates: {sorted(df['missing_rate'].unique())}")
    print(f"  Missing patterns: {list(df['missing_pattern'].unique())}")

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

    print(f"\nPerformance by Missing Pattern:")
    for pattern in df['missing_pattern'].unique():
        pattern_df = df[df['missing_pattern'] == pattern]
        print(f"  {pattern}:")
        print(f"    HMM: Acc={pattern_df['hmm_accuracy'].mean():.3f}, Err={pattern_df['hmm_mean_error'].mean():.1f}c")
        print(f"    FST: Acc={pattern_df['fst_accuracy'].mean():.3f}, Err={pattern_df['fst_mean_error'].mean():.1f}c")

    print(f"\nPerformance by Missing Rate:")
    for rate in sorted(df['missing_rate'].unique()):
        rate_df = df[df['missing_rate'] == rate]
        print(f"  {rate:.1f}:")
        print(f"    HMM: Acc={rate_df['hmm_accuracy'].mean():.3f}, Err={rate_df['hmm_mean_error'].mean():.1f}c")
        print(f"    FST: Acc={rate_df['fst_accuracy'].mean():.3f}, Err={rate_df['fst_mean_error'].mean():.1f}c")

    print(f"\nPerformance by Raga:")
    for raga in df['raga'].unique():
        raga_df = df[df['raga'] == raga]
        print(f"  {raga}:")
        print(f"    HMM: Acc={raga_df['hmm_accuracy'].mean():.3f}, Err={raga_df['hmm_mean_error'].mean():.1f}c")
        print(f"    FST: Acc={raga_df['fst_accuracy'].mean():.3f}, Err={raga_df['fst_mean_error'].mean():.1f}c")

def main():
    """Main function to run the completion evaluation"""
    print("ShrutiSense: Symbolic Music Completion Evaluation Suite")
    print("Task 2: Filling Missing Values in Sequences")
    print("="*80)

    # Initialize evaluator
    evaluator = ComprehensiveCompletionEvaluator()

    # Run evaluation (reduced numbers for demo - increase for full evaluation)
    print("Running completion evaluation...")
    results = evaluator.evaluate_models(
        n_simulations=15,  # Increase to 100+ for full evaluation
        sequence_lengths=[30, 50, 100],
        missing_rates=[0.2, 0.3, 0.4],
        missing_patterns=['random', 'clustered', 'structured']
    )

    # Create visualizations
    visualizer = CompletionResultsVisualizer(results)
    visualizer.create_all_plots()

    # Print statistics and examples
    print_completion_summary_statistics(results)
    print_completion_examples(results)

    # Save results
    with open('completion_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n\nCompletion evaluation complete! Results saved to 'completion_results.json'")
    print("Visualization saved as 'shrutisense_completion_analysis.png'")

if __name__ == "__main__":
    main()
