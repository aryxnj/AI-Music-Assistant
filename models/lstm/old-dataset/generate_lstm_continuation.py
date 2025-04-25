#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import pretty_midi

MODEL_PATH = "lstm_model.h5"
INPUT_MIDI_PATH = "input_midis/input-3.mid"
OUTPUT_MIDI_PATH = "output_midis/output-3-generated.mid"

NUM_GENERATION_STEPS = 64  
TEMPERATURE = 1.0

# Pitch and special tokens
PITCH_LOW, PITCH_HIGH = 21, 108
TOKEN_HOLD = 128
TOKEN_OFF = 129
MAX_TIME_STEP = 0.25  # 1/16th note increments

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

def midi_to_token_sequence(midi_path):
    """
    Simple parser that tries to quantise a monophonic track into 1/16th notes
    with tokens in [21..108, 128=hold, 129=note-off].
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    instrument = None
    for inst in pm.instruments:
        if not inst.is_drum:
            instrument = inst
            break
    if instrument is None or len(instrument.notes) == 0:
        print("No monophonic track found or no notes. Returning empty token sequence.")
        return []

    instrument.notes.sort(key=lambda n: n.start)

    tokens = []
    current_time = 0.0
    note_idx = 0
    current_note = None
    note_off_time = 0.0

    while True:
        if note_idx >= len(instrument.notes) and current_note is None:
            break

        if current_note is None and note_idx < len(instrument.notes):
            nxt_note = instrument.notes[note_idx]
            if nxt_note.start <= current_time:
                current_note = nxt_note
                note_off_time = nxt_note.end
                note_idx += 1
        
        if current_note is not None:
            pitch_val = current_note.pitch
            if pitch_val < PITCH_LOW or pitch_val > PITCH_HIGH:
                pitch_val = max(min(pitch_val, PITCH_HIGH), PITCH_LOW)

            if current_time < note_off_time:
                # Note still sounding => pitch or hold
                if not tokens or tokens[-1] != pitch_val:
                    tokens.append(pitch_val)
                else:
                    tokens.append(TOKEN_HOLD)
            else:
                # The note ended => note_off
                tokens.append(TOKEN_OFF)
                current_note = None
        else:
            tokens.append(TOKEN_OFF)

        current_time += MAX_TIME_STEP

    return tokens

def generate_continuation(model, seed_sequence, num_steps=NUM_GENERATION_STEPS, temperature=TEMPERATURE):
    generated = list(seed_sequence)
    input_seq = tf.expand_dims(generated, 0)

    for _ in range(num_steps):
        logits = model(input_seq)[:, -1, :]
        logits = logits / temperature
        next_token = tf.random.categorical(logits, num_samples=1)
        next_token = tf.squeeze(next_token, axis=-1).numpy()
        generated.append(int(next_token))
        input_seq = tf.expand_dims(generated, 0)
    
    return generated

def token_sequence_to_midi(token_sequence, output_path):
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    pm.instruments.append(instrument)

    current_pitch = None
    note_start_time = 0.0
    current_time = 0.0

    for token in token_sequence:
        if token == TOKEN_HOLD:
            current_time += MAX_TIME_STEP
            continue
        elif token == TOKEN_OFF:
            if current_pitch is not None:
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=current_pitch,
                    start=note_start_time,
                    end=current_time
                )
                instrument.notes.append(note)
                current_pitch = None
            current_time += MAX_TIME_STEP
        else:
            # It's a pitch
            if current_pitch is not None:
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=current_pitch,
                    start=note_start_time,
                    end=current_time
                )
                instrument.notes.append(note)
            current_pitch = token
            note_start_time = current_time
            current_time += MAX_TIME_STEP

    if current_pitch is not None:
        note = pretty_midi.Note(
            velocity=100,
            pitch=current_pitch,
            start=note_start_time,
            end=current_time
        )
        instrument.notes.append(note)

    pm.write(output_path)
    print(f"Saved generated MIDI to {output_path}")

def main():
    # Parse 2-bar input
    seed_tokens = midi_to_token_sequence(INPUT_MIDI_PATH)
    print(f"Original seed token length = {len(seed_tokens)}")

    # This is the “hack” to get 64 steps for your 2-bar melody:
    # If your seed is ~32 tokens, just duplicate.
    # If your seed is 17 tokens or something else, you could do more advanced padding.
    if len(seed_tokens) < 64:
        print("Detected fewer than 64 tokens. Duplicating to reach 64 steps (2 bars -> 4 bars).")
        while len(seed_tokens) < 64:
            seed_tokens = seed_tokens + seed_tokens

        # If we go above 64, slice back down to 64.
        seed_tokens = seed_tokens[:64]
        print(f"Seed token length after duplicating = {len(seed_tokens)}")

    # Generate
    final_sequence = generate_continuation(model, seed_tokens, num_steps=NUM_GENERATION_STEPS, temperature=TEMPERATURE)

    # Convert to MIDI
    token_sequence_to_midi(final_sequence, OUTPUT_MIDI_PATH)

if __name__ == "__main__":
    main()
