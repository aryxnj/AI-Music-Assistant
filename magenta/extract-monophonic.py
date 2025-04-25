import pretty_midi
import os

# Function to check if a MIDI file is monophonic
def is_monophonic(midi_data):
    for instrument in midi_data.instruments:
        # Ensure the instrument contains notes and not a drum (ignore drum tracks)
        if instrument.is_drum:
            continue
        notes = sorted(instrument.notes, key=lambda x: x.start)
        for i in range(1, len(notes)):
            # Check if two notes overlap, meaning it's polyphonic
            if notes[i].start < notes[i-1].end:
                return False
    return True

# Function to preprocess MIDI files
def preprocess_midi_files(midi_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # Create the output directory if it doesn't exist
    for filename in os.listdir(midi_folder):
        if filename.endswith(".mid") or filename.endswith(".midi"):
            filepath = os.path.join(midi_folder, filename)
            try:
                midi_file = pretty_midi.PrettyMIDI(filepath)
                if is_monophonic(midi_file):
                    output_path = os.path.join(output_folder, filename)
                    midi_file.write(output_path)
                    print(f"Processed and saved monophonic file: {filename}")
                else:
                    print(f"Skipped polyphonic file: {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Define the input and output directories
midi_folder = 'training-data/maestro' 
output_folder = 'monophonic-midi'

# Call the preprocessing function
preprocess_midi_files(midi_folder, output_folder)
