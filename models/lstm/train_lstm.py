#!/usr/bin/env python3
import os
import pickle
import numpy as np
import tensorflow as tf

##################################
# 1. Hyperparameters & Constants #
##################################
VOCAB_SIZE = 128  # For notes in [0..127]
EMBED_DIM = 128
RNN_UNITS = 256

BATCH_SIZE = 64
EPOCHS = 10
SHUFFLE_BUFFER_SIZE = 10000

# Paths to your cleaned "Control" pickles
TRAIN_PICKLE = "pickles/train_song_list_cleaned.p"
VAL_PICKLE   = "pickles/validation_song_list_cleaned.p"
TEST_PICKLE  = "pickles/evaluation_song_list_cleaned.p"

##################################
# 2. Load Pickle Data            #
##################################
def load_pickle_data(pickle_path):
    """
    Each .p file should be a Python list of tuples, 
    where each tuple is (X, Y), e.g.:
       X = [n0, n1, n2, ...], 
       Y = [n1, n2, n3, ...]
    Both X and Y are lists of integer pitches in [0..127].
    """
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    return data

def data_generator(data):
    """
    data is a list of melodies;
    each melody is a list of [pitch, duration] pairs.
    We yield (X, Y) for each melody in a next-note format.
    """
    for melody in data:
        # melody is something like [[55, 4], [57, 4], [59, 4], ...]
        if len(melody) < 2:
            continue
        
        # Extract only the pitch from each pair
        pitches = [note[0] for note in melody]
        
        # If there's only 1 pitch, skip it
        if len(pitches) < 2:
            continue
        
        # Next-step prediction
        X = pitches[:-1]  # all but last
        Y = pitches[1:]   # all but first
        
        yield np.array(X, dtype=np.int32), np.array(Y, dtype=np.int32)


def create_dataset(pickle_path, batch_size=BATCH_SIZE, shuffle_buffer=SHUFFLE_BUFFER_SIZE):
    """
    Loads the list of (X, Y) pairs from 'pickle_path' 
    and creates a padded tf.data.Dataset for training.
    """
    data = load_pickle_data(pickle_path)  # List of (x, y)
    # Convert to a generator-based dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(data),
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )
    )

    # Shuffle, pad, batch
    dataset = dataset.shuffle(shuffle_buffer)
    # Since sequences can have variable lengths, we use .padded_batch
    # default padding value = 0, which is still a valid note, but 
    # typically you might want a special <PAD> token. We'll keep it simple here.

    # Take only first 1,000 items to test functionality
    # dataset = dataset.take(100)

    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=([None], [None]),
        padding_values=(0, 0)
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

########################################
# 3. Build the LSTM Model             #
########################################
def build_lstm_model(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, rnn_units=RNN_UNITS):
    """
    Constructs an LSTM-based model for next-note prediction.
    """
    inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)  # variable-length
    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)
    x = tf.keras.layers.LSTM(rnn_units, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(rnn_units, return_sequences=True)(x)
    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

########################################
# 4. Train & Evaluate                  #
########################################
def main():
    # Create datasets
    print("Loading data...")
    train_dataset = create_dataset(TRAIN_PICKLE)
    val_dataset   = create_dataset(VAL_PICKLE)
    test_dataset  = create_dataset(TEST_PICKLE)
    
    # Build model
    print("Building model...")
    model = build_lstm_model()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"]
    )
    
    # Train
    print("Starting training...")
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS
    )
    
    # Evaluate on validation set
    print("Evaluating on validation dataset...")
    val_loss, val_accuracy = model.evaluate(val_dataset)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Evaluate on test set
    print("Evaluating on test dataset...")
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Save the trained model
    model.save("lstm_model.h5")
    print("Model saved to lstm_model.h5")

if __name__ == "__main__":
    main()
