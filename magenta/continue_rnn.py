import os
import tensorflow as tf
from note_seq.protobuf import generator_pb2
from note_seq import midi_io, sequences_lib
from magenta.models.shared import sequence_generator_bundle
from magenta.models.melody_rnn import melody_rnn_sequence_generator

# Directories
models_dir = r'C:\Users\jaina\Downloads\Disss\magenta_models\checkpoints\melody_rnn'
input_midis_dir = r'C:\Users\jaina\Downloads\Disss\input_midis'
output_base_dir = r'C:\Users\jaina\Downloads\Disss\output_midis'

# Create the melody-rnn subfolder inside output_midis
output_dir = os.path.join(output_base_dir, 'melody-rnn')
os.makedirs(output_dir, exist_ok=True)

# Model variants
model_variants = {
    'basic_rnn': 'basic_rnn.mag',
    'lookback_rnn': 'lookback_rnn.mag',
    'attention_rnn': 'attention_rnn.mag',
    'mono_rnn': 'mono_rnn.mag'
}

# Input MIDIs to test
input_files = [f'input-{i}.mid' for i in range(1, 7)]  # input-1.mid ... input-6.mid

# Generation parameters
# Commented line with more extensive parameters (too many outputs)
# temperatures = [0.8, 1.0, 1.2] # multiple temps and step_lengths = [64, 128] would create too many files
# For this run, we choose just two temperatures and one step_length to limit outputs.
temperatures = [0.8, 1.0]  
total_steps = 64            

for model_name, bundle_name in model_variants.items():
    bundle_path = os.path.join(models_dir, bundle_name)
    print(f"\nLoading model: {model_name} from {bundle_path}")
    bundle = sequence_generator_bundle.read_bundle_file(bundle_path)

    generator_map = melody_rnn_sequence_generator.get_generator_map()
    if bundle.generator_details.id not in generator_map:
        print(f"WARNING: Generator id {bundle.generator_details.id} not found for {model_name}. Skipping.")
        continue
    
    generator = generator_map[bundle.generator_details.id](checkpoint=None, bundle=bundle)
    generator.steps_per_quarter = 4  # Set as needed

    for input_file in input_files:
        input_path = os.path.join(input_midis_dir, input_file)
        if not os.path.exists(input_path):
            print(f"Input file {input_path} not found, skipping.")
            continue

        # Load input sequence
        input_sequence = midi_io.midi_file_to_note_sequence(input_path)
        if not input_sequence.time_signatures:
            input_sequence.time_signatures.add(numerator=4, denominator=4)
        if not input_sequence.tempos:
            input_sequence.tempos.add(qpm=120)
        
        input_total_time = input_sequence.total_time

        # Calculate end time for generation based on total_steps
        end_time = input_total_time + (total_steps * 60.0 / (generator.steps_per_quarter * input_sequence.tempos[0].qpm))

        for temperature in temperatures:
            # Create generation request
            generator_options = generator_pb2.GeneratorOptions()
            generator_options.generate_sections.add(
                start_time=input_total_time,
                end_time=end_time
            )
            # Set generation parameters
            generator_options.args['temperature'].float_value = temperature

            print(f"Generating {model_name} on {input_file} with steps={total_steps}, temperature={temperature}")
            generated_sequence = generator.generate(input_sequence, generator_options)

            # Combine sequences
            combined_sequence = sequences_lib.concatenate_sequences([input_sequence, generated_sequence])

            # Construct output filename
            input_stem = os.path.splitext(input_file)[0]  # e.g. "input-1"
            output_filename = f"{input_stem}_{model_name}_temp{temperature}.mid"
            output_path = os.path.join(output_dir, output_filename)

            # Save the combined MIDI
            midi_io.sequence_proto_to_midi_file(combined_sequence, output_path)
            print(f"Saved generated MIDI to {output_path}")
