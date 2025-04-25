import pickle
import random
from collections import defaultdict

def load_data(pickle_path):
    """
    Load the list of melodies from a pickled file.
    Each melody is a list of [pitch, duration] pairs.
    """
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data

def build_markov_chain(melodies):
    """
    Build a first-order Markov chain from the given melodies.
    Each state is a (pitch, duration) tuple, and we track how often
    a state transitions to each subsequent state.

    Returns a dictionary of the form:
        transition_probs[(pitch, duration)] = {
            (next_pitch, next_dur): probability_of_transition,
            ...
        }
    """
    transition_counts = defaultdict(lambda: defaultdict(int))

    # Collect raw counts, converting states to tuples
    for melody in melodies:
        # Convert each note to a tuple (pitch, duration)
        melody_tuples = [tuple(note) for note in melody]
        # Record transitions
        for i in range(len(melody_tuples) - 1):
            current_state = melody_tuples[i]
            next_state = melody_tuples[i + 1]
            transition_counts[current_state][next_state] += 1

    # Convert counts to probabilities
    transition_probs = {}
    for state, next_dict in transition_counts.items():
        total_count = sum(next_dict.values())
        state_probs = {k: v / total_count for k, v in next_dict.items()}
        transition_probs[state] = state_probs

    return transition_probs

if __name__ == "__main__":
    # 1. Load the training data from your pickles folder
    training_data_path = "pickles/train_song_list_cleaned.p"
    melodies = load_data(training_data_path)

    # 2. Build the Markov chain
    transition_probs = build_markov_chain(melodies)

    # 3. Save the model so it can be used later
    model_output_path = "markov_model.pkl"
    with open(model_output_path, "wb") as f:
        pickle.dump(transition_probs, f)

    print(f"Model trained and saved to {model_output_path}")
