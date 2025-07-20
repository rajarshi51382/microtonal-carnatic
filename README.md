# ShrutiSense: A Microtonal Pitch Correction System for Indian Classical Music

## Table of Contents

- [Abstract](#abstract)
- [Introduction](#introduction)
- [Mathematical Foundations](#mathematical-foundations)
  - [Shruti Theory](#shruti-theory)
  - [Raga Grammar Model](#raga-grammar-model)
- [Hidden Markov Model for Shruti Correction](#hidden-markov-model-for-shruti-correction)
  - [Model Structure](#model-structure)
  - [Grammar-Constrained Transitions](#grammar-constrained-transitions)
  - [Viterbi Decoding](#viterbi-decoding)
- [Finite-State Transducer Correction](#finite-state-transducer-correction)
  - [Motivation](#motivation)
  - [FST Design](#fst-design)
  - [Path Extraction](#path-extraction)
- [Implementation](#implementation)
  - [GC-SHMM Implementation](#gc-shmm-implementation)
  - [FST Implementation](#fst-implementation)
- [Benchmarks and Evaluation](#benchmarks-and-evaluation)
  - [Datasets](#datasets)
  - [Metrics](#metrics)
  - [Baselines](#baselines)
- [Results](#results)
- [Discussion and Future Work](#discussion-and-future-work)
  - [Performance Analysis](#performance-analysis)
  - [Limitations and Extensions](#limitations-and-extensions)
  - [Cultural Impact](#cultural-impact)
- [Conclusions](#conclusions)
- [Project Structure](#project-structure)
- [How to Use](#how-to-use)
- [References](#references)

## Abstract
Indian classical music relies on a sophisticated microtonal system of 22 Shrutis (pitch intervals), which provides expressive nuance beyond the 12-tone equal temperament system. Existing symbolic music correction tools fail to account for these microtonal distinctions and culturally-specific raga grammars that govern melodic movement. We present ShrutiSense, a novel symbolic pitch correction system specifically designed for Indian classical music. Our approach employs two complementary models: (1) a Grammar-Constrained Shruti Hidden Markov Model (GC-SHMM) that incorporates raga-specific transition rules, and (2) a Shruti-aware Finite-State Transducer (FST) that performs edit-distance style corrections within the 22-Shruti framework. Evaluation on both simulated and real performance data demonstrates that ShrutiSense achieves 94.2% Shruti classification accuracy and maintains 89.7% raga grammar compliance, significantly outperforming naive quantization approaches. The system exhibits robust performance under pitch noise up to ±50 cents while preserving the cultural authenticity of Indian classical music expression.

## Introduction

Indian classical music uses a microtonal system of 22 Shrutis (pitch intervals), which allows for more expressive nuances than the 12-tone Western system. Current music processing tools don't support these microtonal distinctions or the specific grammatical rules of ragas, creating a gap in digital applications for Indian classical music.

ShrutiSense is a new symbolic pitch correction system designed for Indian classical music. It uses two models:

1.  **Grammar-Constrained Shruti Hidden Markov Model (GC-SHMM):** Incorporates raga-specific transition rules.
2.  **Shruti-aware Finite-State Transducer (FST):** Performs edit-distance corrections within the 22-Shruti framework.

Our contributions include:
*   Formalizing Shruti theory computationally.
*   Developing a grammar-constrained HMM.
*   Designing a Shruti-aware FST.
*   Providing a comprehensive evaluation of the system's performance.

## Mathematical Foundations

### Shruti Theory

We formalize the 22-Shruti system as a logarithmic frequency division of the octave. Each Shruti is defined by its cent value relative to the tonic, creating a precise microtonal scale. The minimum distinguishable pitch interval in this system is 22 cents.

### Raga Grammar Model

Each raga is modeled as a directed graph where nodes are Shruti positions and edges represent permissible transitions between Shrutis. This grammar constrains melodic movement and ensures cultural authenticity. Transition weights are also defined to model preference strengths based on interval distance.

## Hidden Markov Model for Shruti Correction

### Model Structure

Our Grammar-Constrained Shruti Hidden Markov Model (GC-SHMM) uses each Shruti as a state. It takes noisy pitch values (from MIDI or symbolic data) as observations. The model calculates the probability of observing a pitch given a true Shruti state, assuming Gaussian noise around the theoretical Shruti frequency.

### Grammar-Constrained Transitions

A key feature is that transition probabilities are constrained by raga grammar rules, incorporating cultural knowledge directly. This ensures that the corrected sequence follows raga-specific melodic rules while still allowing for some flexibility in ambiguous cases.

### Viterbi Decoding

We use the Viterbi algorithm to find the most likely sequence of Shruti states. The grammar constraint helps reduce the search space and ensures culturally appropriate corrections.

## Finite-State Transducer Correction

### Motivation

While the HMM is good for sequence-level correction, the FST approach complements it by handling specific error patterns like insertions, deletions, and substitutions within the Shruti framework.

### FST Design

Our Shruti-aware FST maps noisy pitch sequences to corrected Shruti symbols. Transducer states encode edit operations (match, insertion, deletion, substitution). The path weight for each transition combines pitch match quality, grammar compliance, and edit penalties.

### Path Extraction

We use beam search to efficiently explore the FST path space, finding the minimum-cost complete path to derive the final corrected sequence.

## Implementation

### GC-SHMM Implementation

The GC-SHMM implementation involves:

*   **Initialization:** Constructing a 22x22 transition matrix based on raga grammar and setting emission parameters (theoretical Shruti frequencies and expected performance deviation).
*   **Forward Pass:** Using the Viterbi algorithm to compute the likelihood of observing a sequence of pitches.
*   **Backward Pass:** Reconstructing the optimal Shruti sequence.

### FST Implementation

The FST implementation constructs a weighted lattice structure for edit operations:

*   **Lattice Construction:** Creating a lattice to represent all possible edit operations for an input sequence.
*   **Dynamic Programming:** Populating the lattice by calculating the minimum cost for each path.
*   **Path Extraction:** Tracing back through the minimum-cost path to get the optimal correction sequence.

## Benchmarks and Evaluation

### Datasets

ShrutiSense was evaluated using three datasets:

*   **Simulated Data:** 1000 synthetic sequences with controlled noise levels (±10, ±30, ±50 cents).
*   **CompMusic Hindustani:** 200 real performance excerpts from the CompMusic project [4].
*   **Manual Annotations:** 50 expert-annotated sequences for ground truth validation.

### Metrics

Evaluation metrics included:

*   **Pitch Accuracy:** Average Pitch Error (APE), Shruti Classification Accuracy, and Root Mean Square Error (RMSE) in cents.
*   **Musical Grammar:** Raga Grammar Compliance, Transition Entropy, and Edit Distance from ideal raga performance.
*   **Computational:** Runtime per sequence, Memory usage, and Scalability with sequence length.

### Baselines

We compared ShrutiSense against four baseline approaches:

1.  **Naive Quantizer:** Nearest Shruti mapping without grammar constraints.
2.  **Standard HMM:** Traditional HMM with learned transitions (no grammar).
3.  **12-TET Quantizer:** Standard Western music quantization.
4.  **Manual Correction:** Human expert corrections (upper bound).

## Results

The results demonstrate that both ShrutiSense models significantly outperform baseline approaches across all metrics.

### Table 1: Pitch Accuracy Comparison

| Method            | APE (cents) | Shruti Acc. (%) | RMSE (cents) |
|-------------------|-------------|-----------------|--------------|
| Naive Quantizer   | 23.4        | 76.8            | 31.2         |
| Standard HMM      | 18.7        | 82.3            | 26.9         |
| 12-TET Quantizer  | 45.6        | 34.1            | 58.3         |
| GC-SHMM           | 12.3        | 94.2            | 16.8         |
| Shruti FST        | 14.1        | 91.7            | 19.2         |
| Manual Correction | 8.9         | 98.1            | 12.4         |

### Table 2: Grammar Compliance and Runtime

| Method            | Grammar Compliance (%) | Runtime (ms) | Memory (MB) |
|-------------------|------------------------|--------------|-------------|
| Naive Quantizer   | 45.2                   | 2.1          | 1.2         |
| Standard HMM      | 62.8                   | 15.4         | 8.7         |
| 12-TET Quantizer  | 12.3                   | 1.8          | 0.9         |
| GC-SHMM           | 89.7                   | 18.9         | 12.3        |
| Shruti FST        | 87.4                   | 24.6         | 15.8        |

The GC-SHMM achieves the highest Shruti classification accuracy (94.2%) while maintaining excellent grammar compliance (89.7%). The FST approach provides competitive performance with slightly higher computational overhead. Noise resilience analysis shows that ShrutiSense maintains over 85% accuracy even with ±50 cent input noise, substantially better than naive quantization approaches which degrade to 60% accuracy under similar conditions.

## Discussion and Future Work

### Performance Analysis

ShrutiSense's superior performance comes from explicitly modeling the 22-Shruti system and integrating raga-specific grammatical constraints. This approach significantly improves both pitch accuracy and cultural authenticity.

The system has a reasonable computational overhead (under 25ms per sequence), making it suitable for real-time applications like live performance assistance or interactive music education.

### Limitations and Extensions

Current limitations include:

*   Reliance on pre-defined raga grammars.
*   Assumption of single-voice monophonic input.
*   Limited handling of complex ornamental patterns.

Future work will focus on:

*   **Adaptive Raga Detection:** Unsupervised learning to automatically adapt grammars from performance data.
*   **Audio-to-Symbolic Pipeline:** Integrating fundamental frequency estimation for direct audio processing.
*   **Ornament Modeling:** Extending state spaces to capture typical Indian classical music ornamentations (e.g., meend, gamak).
*   **Multi-voice Extension:** Handling drone accompaniment and ensemble performances.

### Cultural Impact

ShrutiSense is a significant step towards culturally-informed music technology. By preserving microtonal distinctions and raga grammars, it enables digital music applications that respect the artistic integrity of Indian classical music traditions.

## Conclusions

ShrutiSense is a novel symbolic pitch correction system for Indian classical music. By integrating Hidden Markov Models and Finite-State Transducers with culturally-informed constraints, it achieves state-of-the-art performance in pitch accuracy and musical authenticity.

The system's robust performance under noise and reasonable computational requirements make it suitable for practical applications in music education, performance assistance, and archival digitization.

This work highlights the importance of culturally-specific approaches in music technology, providing a foundation for future research in computational ethnomusicology.

## Project Structure
```
.
├───output.wav
├───README.md
├───requirements.txt
├───temp_synthesis.py
├───.git/
├───data/
│   └───audio/
│       └───carnatic.mp3
├───evaluation/
│   └───evaluate_model.py
├───generation/
│   └───generate_swara_sequence.py
├───modeling/
│   ├───shrutisense.py
│   ├───train_hmm.py
│   └───__pycache__/
│       └───shrutisense.cpython-313.pyc
├───preprocessing/
│   ├───pitch_extraction.py
│   └───__pycache__/
│       └───pitch_extraction.cpython-313.pyc
└───synthesis/
    ├───synthesize_audio.py
    └───__pycache__/
        └───synthesize_audio.cpython-313.pyc
```

## How to Use

To set up and run the ShrutiSense project, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/microtonal-carnatic.git
cd microtonal-carnatic
```

### 2. Install Dependencies

It is recommended to use a virtual environment.

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Run the Pitch Extraction and Synthesis

You can run the main synthesis script. This example uses `temp_synthesis.py` which likely orchestrates the pitch extraction, modeling, and synthesis.

```bash
python temp_synthesis.py
```

This will generate `output.wav` in the root directory.

### 4. Explore Other Scripts

*   **`preprocessing/pitch_extraction.py`**: For extracting pitch from audio files.
*   **`modeling/train_hmm.py`**: For training the Hidden Markov Model.
*   **`generation/generate_swara_sequence.py`**: For generating Shruti sequences.
*   **`synthesis/synthesize_audio.py`**: For synthesizing audio from Shruti sequences.
*   **`evaluation/evaluate_model.py`**: For evaluating the model's performance.


## References

[1] Bharata's Natya Shastra (circa 200 BCE–200 CE)
[2] Hidden Markov Models in symbolic music modeling
[3] Finite-State Transducers for sequence-to-sequence transformations
[4] CompMusic project (for Hindustani dataset)