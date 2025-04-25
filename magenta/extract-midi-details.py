from note_seq import midi_io
import pretty_midi


# Path to the input MIDI file
input_midi_path = r'C:\Users\jaina\Downloads\Disss\input_midis\test.mid'

# Load the MIDI file
note_sequence = midi_io.midi_file_to_note_sequence(input_midi_path)

# Print the NoteSequence details
print("Ticks per quarter:", note_sequence.ticks_per_quarter)
print("Time signatures:", note_sequence.time_signatures)
print("Tempos:", note_sequence.tempos)
print("Number of notes:", len(note_sequence.notes))

# Print all notes
for note in note_sequence.notes:
    print(f"Pitch: {note.pitch}, Start: {note.start_time}, End: {note.end_time}, Velocity: {note.velocity}")

if len(note_sequence.notes) == 0:
    print("The MIDI file is empty or contains no notes.")
else:
    print("MIDI file contains valid notes.")

midi_file = pretty_midi.PrettyMIDI(r'C:\Users\jaina\Downloads\Disss\input_midis\test.mid')

# Print tracks
print("Number of instruments:", len(midi_file.instruments))
for instrument in midi_file.instruments:
    print(f"Instrument: {instrument.name}, Program: {instrument.program}, Notes: {len(instrument.notes)}")

# Print notes
for instrument in midi_file.instruments:
    print(f"Notes for {instrument.name}:")
    for note in instrument.notes:
        print(f"Pitch: {note.pitch}, Start: {note.start}, End: {note.end}, Velocity: {note.velocity}")

if not midi_file.instruments or all(len(instr.notes) == 0 for instr in midi_file.instruments):
    print("The MIDI file is empty or contains no notes.")
else:
    print("MIDI file contains valid tracks and notes.")
