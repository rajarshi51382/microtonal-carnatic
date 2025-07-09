# Carnatic Microtonal Expression Pipeline  
**Authors:** Rajarshi Ghosh and Jayanth Athipatla

## Overview  
This project generates expressive, microtonally-inflected performances of Carnatic melodies. It takes a sequence of symbolic Swaras (notes) and outputs audio that reflects the stylistic nuances of a chosen Raga, including **Gamaka** (glides, oscillations, and pitch inflections).

---

## Step 1: Collect and Preprocess Training Data

**Input:**  
High-quality audio recordings of Carnatic music with Raga labels.

**Tasks:**
- Curate a dataset of Carnatic performances across multiple Ragas.
- Normalize pitch (e.g., fix tonic to **C4 = 261.63 Hz**).
- Use pitch tracking tools to extract continuous pitch contours:
  - `librosa.piptrack`
  - `crepe`
  - `yin`
- Align pitch contours with symbolic Swara annotations (if available).
- Segment pitch contours into note regions and compute:
  - **Base Swara** (nearest note in just intonation)
  - **Microtonal deviation** (in cents)
  - *Optional:* **Gamaka type** (flat, glide, oscillation)

---

## Step 2: Model Microtonal Deviations with HMMs

**Option A: Per-Note-Class HMMs**
- Train a **Gaussian HMM** for each Swara (Sa, Re, Ga, etc.).
- Capture expressive variation within each Swara class.

**Option B: Sequence-Level HMM**
- Model sequences of (Swara, deviation) pairs with a unified HMM.
- Use hidden states to represent expressive contexts (e.g., “approach to Ga”, “nyas note”).

**Implementation:**
- Use [`hmmlearn`](https://hmmlearn.readthedocs.io/) for modeling.
- Input: sequences of Swaras and their microtonal deviations.
- Output: temporal dynamics of expressive pitch behavior.

---

## Step 3: Generate Base Swara Sequences

**Input:**  
A melody generation model (e.g., Markov model, Transformer).

**Tasks:**
- Generate Swara sequences for a specific Raga.
- Enforce **Arohana/Avarohana** (ascending/descending scale) constraints.
- Incorporate **Raga-specific phrases and motifs** to maintain authenticity.

---

## Step 4: Add Microtonal Expression

For each generated note:
- Identify its **Swara class**.
- Use the trained HMM (or generative model) to sample:
  - **Microtonal deviation** (in cents)
  - *Optional:* a full **pitch trajectory** over time
- Combine the symbolic note with expressive deviations to yield **realistic ornamentation**.

---

## Step 5: Synthesize Output

**Option A: MIDI with Pitch Bend**
- Convert the Swara + deviation output into MIDI.
- Use **pitch bend messages** for fine microtonal tuning.
- Render with a software synth that supports pitch bends (e.g., FluidSynth, Kontakt).

**Option B: Direct Audio Synthesis**
- Use tools like `pyo`, `scamp`, or `csound`.
- Interpolate pitch smoothly to implement **Gamakas** and expressive motion.

---

## Step 6: Evaluate

**Automatic Evaluation**
- Compare pitch deviation distributions with real Carnatic performances.
- Analyze alignment with expected **Raga microtonal structures**.

**Human Evaluation (LATER)**
- Collect feedback from expert musicians and listeners.
- Evaluate:
  - **Raga conformity**
  - **Aesthetic quality**
  - **Expressive realism**

---

## Step 7: Extend (Optional)

- Model **Gamaka trajectories** as reusable templates.
- Add **rhythmic conditioning** based on the Tala cycle.
- Explore advanced neural models:
  - **Mixture Density Networks (MDNs)**
  - **Transformers with continuous outputs**
  - **Diffusion-based pitch models**

---

## Project Structure

```
microtonal-carnatic/
├── data/
│   ├── audio/                 
│   ├── annotations/           
├── preprocessing/
│   └── pitch_extraction.py
├── modeling/
│   └── train_hmm.py
├── generation/
│   └── generate_swara_sequence.py
├── synthesis/
│   └── synthesize_audio.py
├── evaluation/
│   └── evaluate_model.py
├── README.md
```

---

## Dependencies

- Python 3.8+  
- Libraries:
  - `librosa`
  - `crepe`
  - `hmmlearn`
  - `numpy`
  - `scipy`
  - `mido`
  - `pyo`
  - `matplotlib`

Install with:

```bash
pip install librosa crepe hmmlearn mido pyo matplotlib
```

---

## Citation

If you use or extend this pipeline, please cite:

**Rajarshi Ghosh and Jayanth Athipatla**, *Carnatic Microtonal Expression Pipeline*, 2025.  
