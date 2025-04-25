import os
import numpy as np
import tensorflow as tf
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
from note_seq import midi_io
from note_seq.protobuf import music_pb2

# Suppress TensorFlow GPU warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Paths
checkpoint_dir = r'C:\Users\jaina\Downloads\Disss\magenta_models\checkpoints\hierdec-mel_16bar'
checkpoint_file = os.path.join(checkpoint_dir, 'hierdec-mel_16bar.ckpt')
input_midi_path = r'C:\Users\jaina\Downloads\Disss\input_midis\test.mid'
output_dir = r'C:\Users\jaina\Downloads\Disss\output_midis'
output_midi_path = os.path.join(output_dir, 'test_cont.mid')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Model configuration
config_name = 'hierdec-mel_16bar'
config = configs.CONFIG_MAP[config_name]

# Verify checkpoint_file exists
if not os.path.exists(checkpoint_file + '.index'):
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

# Initialize the model with the checkpoint file
model = TrainedModel(config, batch_size=8, checkpoint_dir_or_path=checkpoint_file)

# Load the MIDI file as a NoteSequence
input_sequence = midi_io.midi_file_to_note_sequence(input_midi_path)

# Validate input MIDI file
if len(input_sequence.notes) == 0:
    raise ValueError("Input MIDI file contains no notes.")

# Debug: Inspect the input sequence
print(f"Input sequence contains {len(input_sequence.notes)} notes.")

# Add time signature and tempo if missing
if not input_sequence.time_signatures:
    input_sequence.time_signatures.add(numerator=4, denominator=4)
if not input_sequence.tempos:
    input_sequence.tempos.add(qpm=120)  # Default tempo: 120 BPM

# Generate continuation using sampling
sequence_length = 256  # 16 bars
generated_sequences = model.sample(
    n=1,
    length=sequence_length,
    temperature=1.0
)

# Take the first generated sequence
generated_sequence = generated_sequences[0]

# Debug: Inspect the generated sequence
print(f"Generated sequence contains {len(generated_sequence.notes)} notes.")

# Concatenate the original input sequence and the generated sequence
combined_sequence = music_pb2.NoteSequence()

# Copy input sequence into the combined sequence
for note in input_sequence.notes:
    new_note = combined_sequence.notes.add()
    new_note.CopyFrom(note)

# Copy time signatures and tempos
combined_sequence.time_signatures.extend(input_sequence.time_signatures)
combined_sequence.tempos.extend(input_sequence.tempos)

# Adjust the timing of the generated sequence to come after the input sequence
offset_time = input_sequence.total_time
for note in generated_sequence.notes:
    new_note = combined_sequence.notes.add()
    new_note.pitch = note.pitch
    new_note.start_time = note.start_time + offset_time
    new_note.end_time = note.end_time + offset_time
    new_note.velocity = note.velocity

# Update the total time of the combined sequence
combined_sequence.total_time = offset_time + generated_sequence.total_time

# Save the combined sequence as a MIDI file
midi_io.sequence_proto_to_midi_file(combined_sequence, output_midi_path)
print(f"Generated continuation with original sequence saved to: {output_midi_path}")
