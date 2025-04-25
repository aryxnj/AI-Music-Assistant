#!/usr/bin/env python3
import tensorflow as tf
import os

####################
# Hyperparameters  #
####################

# You mentioned the dataset uses pitches in the range [21..108] plus 128=hold and 129=note-off.
# That means the largest possible index encountered is 129.
# Therefore, set VOCAB_SIZE so it covers [0..129], i.e. 130 distinct values.
VOCAB_SIZE = 130  # Now covers all values up to 129.
EMBED_DIM = 128
RNN_UNITS = 256

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 10000
EPOCHS = 10

##################################
# 1. Parsing the TFRecord Files  #
##################################
def parse_tfrecord_fn(serialised_example):
    """
    Parse each TFRecord SequenceExample to obtain (input_seq, target_seq).
    Each melody is 64 steps; we create an offset of 1 for next-step prediction.
    """
    context_features = {}
    sequence_features = {
        "pitch_seq": tf.io.VarLenFeature(tf.int64)
    }
    
    _, sequence_data = tf.io.parse_single_sequence_example(
        serialised_example,
        context_features=context_features,
        sequence_features=sequence_features
    )
    
    # Convert the sparse pitch_seq into a dense tensor
    pitch_seq = tf.sparse.to_dense(sequence_data["pitch_seq"])
    
    # Ensure pitch_seq is 1D (shape: (64,))
    pitch_seq = tf.reshape(pitch_seq, [-1])  # e.g., shape: (64,)

    # For next-step prediction: input_seq = pitch_seq[:-1], target_seq = pitch_seq[1:]
    input_seq = pitch_seq[:-1]
    target_seq = pitch_seq[1:]
    
    return input_seq, target_seq

def create_dataset(tfrecord_pattern, batch_size=BATCH_SIZE, shuffle_buffer=SHUFFLE_BUFFER_SIZE):
    """
    Create a TensorFlow Dataset from TFRecord files matching tfrecord_pattern.
    """
    files = tf.io.gfile.glob(tfrecord_pattern)
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Take only the first 1,000 examples to test functionality
    dataset = dataset.take(10000)

    dataset = dataset.shuffle(shuffle_buffer).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

################################
# 2. Build the LSTM Model      #
################################
def build_lstm_model(vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, rnn_units=RNN_UNITS):
    inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)  # variable-length sequences
    
    # Embedding
    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)
    
    # LSTM layers
    x = tf.keras.layers.LSTM(rnn_units, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(rnn_units, return_sequences=True)(x)
    
    # Output layer for next-token prediction
    outputs = tf.keras.layers.Dense(vocab_size, activation="softmax")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

########################################
# 3. Train/Evaluate the Model          #
########################################
def main():
    # File patterns (relative to current directory "LSTM")
    train_pattern = os.path.join("13369389", "4_bars_melodies_pitchseq_train", "train_pitchseq-*-of-00008.tfrecord")
    val_pattern = os.path.join("13369389", "4_bars_melodies_pitchseq_validation", "validation_pitchseq-*-of-00008.tfrecord")
    test_pattern = os.path.join("13369389", "4_bars_melodies_pitchseq_test", "test_pitchseq-*-of-00008.tfrecord")
    
    # Create Datasets
    train_dataset = create_dataset(train_pattern)
    val_dataset = create_dataset(val_pattern)
    test_dataset = create_dataset(test_pattern)
    
    # Build model
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
