import pretty_midi
import numpy as np
from scipy.stats import entropy
from fastdtw import fastdtw

# Calculate Pitch Entropy for melodic unpredictability
def calculate_pitch_entropy(midi_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    notes = [note.pitch for instr in midi_data.instruments for note in instr.notes]
    if not notes:
        return 0.0
    pitch_counts = np.bincount(notes, minlength=128)
    pitch_probabilities = pitch_counts / pitch_counts.sum()
    pitch_entropy = -np.sum(
        pitch_probabilities[pitch_probabilities > 0] * 
        np.log2(pitch_probabilities[pitch_probabilities > 0])
    )
    return pitch_entropy

# Extract rhythm intervals (in beats) rather than seconds
def extract_rhythm_intervals_in_beats(midi_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    beats = midi_data.get_beats()  # Get the beat times
    instrument_notes = sorted(
        [note for instr in midi_data.instruments for note in instr.notes],
        key=lambda note: note.start
    )
    
    if len(beats) < 2:
        return np.array([])  # Avoid division by zero in DTW

    # Map onset times to their nearest beat
    onset_beats = np.array([
        min(beats, key=lambda b: abs(b - note.start)) for note in instrument_notes
    ])

    rhythm_intervals = np.diff(onset_beats)  # Compute time between note onsets
    return rhythm_intervals

# DTW implementation with normalization
def rhythmic_similarity(primer_path, continuation_path):
    primer_intervals = extract_rhythm_intervals_in_beats(primer_path)
    continuation_intervals = extract_rhythm_intervals_in_beats(continuation_path)
    
    if len(primer_intervals) == 0 or len(continuation_intervals) == 0:
        return 0.0

    # Calculate DTW distance between rhythm sequences
    distance, _ = fastdtw(primer_intervals, continuation_intervals)
    
    # Normalize similarity to [0, 1]
    rhythmic_similarity_score = np.exp(-distance / len(primer_intervals))
    
    return rhythmic_similarity_score

# Define primer files and models
primer_indices = [3, 4, 6]
primer_midi_files = { idx: f"testing_inputs/input-{idx}.mid" for idx in primer_indices }

continuation_models = {
    "LSTM":      "lstm",
    "Attention": "attention",
    "Basic":     "basic",
    "Lookback":  "lookback",
    "Mono":      "mono",
    "Markov":    "markov",
}

# Store results for each model across primer inputs
results = {model: {"Pitch Entropy": [], "Rhythmic Similarity": []} for model in continuation_models}

# Loop over each primer file and corresponding continuation files
for idx in primer_indices:
    primer_path = primer_midi_files[idx]
    for model_name, suffix in continuation_models.items():
        continuation_path = f"testing_inputs/output-{idx}-{suffix}.mid"
        entropy_val = calculate_pitch_entropy(continuation_path)
        rhythmic_sim = rhythmic_similarity(primer_path, continuation_path)
        results[model_name]["Pitch Entropy"].append(entropy_val)
        results[model_name]["Rhythmic Similarity"].append(rhythmic_sim)

# Print the average metrics and sample standard deviations for each model
print("Metrics over Input-3, Input-4, and Input-6:")
for model, metrics in results.items():
    entropy_vals = metrics["Pitch Entropy"]
    rhythm_vals = metrics["Rhythmic Similarity"]
    
    avg_entropy = np.mean(entropy_vals) if entropy_vals else 0
    std_entropy = np.std(entropy_vals, ddof=1) if len(entropy_vals) > 1 else 0
    
    avg_rhythm = np.mean(rhythm_vals) if rhythm_vals else 0
    std_rhythm = np.std(rhythm_vals, ddof=1) if len(rhythm_vals) > 1 else 0
    
    print(f"Model: {model}")
    print(f"  Average Pitch Entropy: {avg_entropy:.3f}")
    print(f"  Sample Std Dev of Pitch Entropy: {std_entropy:.3f}")
    print(f"  Average Rhythmic Similarity (DTW): {avg_rhythm:.3f}")
    print(f"  Sample Std Dev of Rhythmic Similarity: {std_rhythm:.3f}\n")
