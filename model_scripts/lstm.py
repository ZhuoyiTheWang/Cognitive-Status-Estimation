import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model
from keras.layers import LSTM, Dense, Masking, Input, Concatenate, Dropout, Conv1D, MaxPooling1D
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

# Specify the directory path
directory = Path('data/Master Dataset')

# List all files in the directory
all_files = [f for f in directory.rglob('*') if f.is_file()]
trials = []

# Process the list of files
for file in all_files:
    file = str(file)
    file = file.split('\\')[2].split('.')[0]
    trials.append(file)

# Initialize lists to store all training samples and their metadata
all_training_samples = []
metadata = []

for trial in trials:
    curr_data = pd.read_csv(f'data/Master Dataset/{trial}.csv', dtype=str)
    curr_data = curr_data.drop_duplicates()

    # Group the data by 'Quadrant' and 'Object'
    grouped = curr_data.groupby(['Quadrant', 'Object'])

    # Iterate over each group (quadrant and object)
    for (quadrant, object_name), object_data in grouped:
        # Convert 'Utterance' to numerical values for proper sorting
        object_data['Utterance'] = object_data['Utterance'].astype(int)
        # Sort by numerical order of 'Utterance'
        object_data = object_data.sort_values(by='Utterance')

        # Drop duplicates per 'Utterance', keeping the first occurrence
        object_data = object_data.drop_duplicates(subset='Utterance', keep='first')

        # Define the default previous info for Utterance 0
        default_prev_info = {
            'Utterance': 0,
            'Current Status': 'F' if object_name.split('-')[0] == ('Q' + str(quadrant)) else "UI",
            'Mentioned': 'N',
            'Gesture': 'N',
            'Grammatical Role': ''
        }

        # Loop through all utterances to create training samples
        # Each iteration (i) corresponds to one utterance. i is zero-based.
        for i in range(len(object_data)):
            if i == 0:
                # For the first utterance, include Utterance 0
                input_data = [default_prev_info]
            else:
                # For subsequent utterances, include all previous utterances
                input_data = object_data.iloc[:i][['Utterance', 'Current Status', 'Mentioned', 'Gesture', 'Grammatical Role']].to_dict(orient='records')

            # Next features for the current utterance
            next_features = object_data.iloc[i][['Mentioned', 'Gesture', 'Grammatical Role']].to_dict()
            # Target: Current Status of the current utterance
            target = object_data.iloc[i]['Current Status']

            # Create the sample
            sample = {
                'quadrant': quadrant,
                'object': object_name,
                'input_data': input_data,
                'next_features': next_features,
                'target': target,
                'trial': trial
            }
            all_training_samples.append(sample)

            # Store metadata including the utterance index (1-based)
            # i goes from 0 to (num_utterances-1), so utterance_index = i+1
            if i == 14:
                metadata.append({
                    'trial': trial,
                    'quadrant': quadrant,
                    'object': object_name,
                    'utterance_index': i+1,
                    'input_data': object_data.iloc[:15][['Utterance', 'Current Status', 'Mentioned', 'Gesture', 'Grammatical Role']].to_dict(orient='records'),
                })
            else:
                metadata.append({
                    'trial': trial,
                    'quadrant': quadrant,
                    'object': object_name,
                    'utterance_index': i+1,
                    'input_data': input_data,
                })

# Initialize label encoders
status_encoder = LabelEncoder()
mentioned_encoder = LabelEncoder()
gesture_encoder = LabelEncoder()
grammar_role_encoder = LabelEncoder()

# Fit encoders on the entire dataset
status_encoder.fit([utterance['Current Status'] for sample in all_training_samples for utterance in sample['input_data']] +
                   [sample['target'] for sample in all_training_samples])
mentioned_encoder.fit([utterance['Mentioned'] for sample in all_training_samples for utterance in sample['input_data']] +
                      [sample['next_features']['Mentioned'] for sample in all_training_samples])
gesture_encoder.fit([utterance['Gesture'] for sample in all_training_samples for utterance in sample['input_data']] +
                    [sample['next_features']['Gesture'] for sample in all_training_samples])
grammar_role_encoder.fit([utterance['Grammatical Role'] for sample in all_training_samples for utterance in sample['input_data']] +
                    [sample['next_features']['Grammatical Role'] for sample in all_training_samples])

# Put them in a dictionary
encoders_dict = {
    'status_encoder': status_encoder,
    'mentioned_encoder': mentioned_encoder,
    'gesture_encoder': gesture_encoder,
    'grammar_role_encoder': grammar_role_encoder
}

# Save to a pkl file
with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders_dict, f)

# Prepare input sequences, next features, and targets
X_sequences = []
X_next_features = []
y = []
sequence_lengths = []

for sample in all_training_samples:
    input_sequence = [
        [
            utterance['Utterance'],
            status_encoder.transform([utterance['Current Status']])[0],
            mentioned_encoder.transform([utterance['Mentioned']])[0],
            gesture_encoder.transform([utterance['Gesture']])[0],
            grammar_role_encoder.transform([utterance['Grammatical Role']])[0]
        ]
        for utterance in sample['input_data']
    ]

    sequence_lengths.append(len(input_sequence))

    next_feats_encoded = [
        mentioned_encoder.transform([sample['next_features']['Mentioned']])[0],
        gesture_encoder.transform([sample['next_features']['Gesture']])[0],
        grammar_role_encoder.transform([sample['next_features']['Grammatical Role']])[0]
    ]
    target = status_encoder.transform([sample['target']])[0]

    X_sequences.append(input_sequence)
    X_next_features.append(next_feats_encoded)
    y.append(target)

X_sequences_padded = pad_sequences(X_sequences, padding='post', dtype='float32')
X_next_features = np.array(X_next_features)
y = np.array(y)
sequence_lengths = np.array(sequence_lengths)

# Define input layers
sequence_input = Input(shape=(X_sequences_padded.shape[1], X_sequences_padded.shape[2]))
next_features_input = Input(shape=(X_next_features.shape[1],))

# conv = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(sequence_input)
# conv = MaxPooling1D(pool_size=2)(conv)

# LSTM layers for sequence data
x = LSTM(128, return_sequences=True)(sequence_input)
x = Dropout(0.1)(x)

for i in range(1):
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.1)(x)

x = LSTM(64)(x)
x = Dropout(0.1)(x)

# Concatenate LSTM output with next features
concat = Concatenate()([x, next_features_input])

# Fully connected layers
fc = Dense(64, activation='relu')(concat)
fc = Dropout(0.1)(fc)
fc = Dense(32, activation='relu')(fc)
fc = Dropout(0.1)(fc)

# Output layer
output = Dense(len(status_encoder.classes_), activation='softmax')(fc)

checkpoint_path = 'best_model.keras'
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping_callback = EarlyStopping(
    monitor='val_accuracy',
    patience=40,
    restore_best_weights=True,
    verbose=1,
    mode='max'
)

model = Model(inputs=[sequence_input, next_features_input], outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

plot_model(model, to_file='LSTM_architecture.png', show_shapes=True, show_layer_names=True)

# Split data into training and testing sets, include metadata in the split so we know which go to test
X_seq_train, X_seq_test, X_next_train, X_next_test, y_train, y_test, seq_lengths_train, seq_lengths_test, metadata_train, metadata_test = train_test_split(
    X_sequences_padded, X_next_features, y, sequence_lengths, metadata, test_size=0.2, random_state=42
)

history = model.fit(
    [X_seq_train, X_next_train],
    y_train,
    epochs=200,
    batch_size=16,
    validation_split=0.1,
    verbose=1,
    callbacks=[model_checkpoint_callback, early_stopping_callback]
)

model = tf.keras.models.load_model(checkpoint_path)

test_loss, test_accuracy = model.evaluate([X_seq_test, X_next_test], y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy:.4f}")

y_pred_probs = model.predict([X_seq_test, X_next_test])
y_pred_classes = np.argmax(y_pred_probs, axis=1)

y_test_labels = status_encoder.inverse_transform(y_test)
y_pred_labels = status_encoder.inverse_transform(y_pred_classes)

desired_ordered_labels = ['UI', 'F', 'A', 'IF']
conf_matrix = confusion_matrix(
    y_test_labels, y_pred_labels, labels=desired_ordered_labels
)
disp = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix, display_labels=desired_ordered_labels
)
plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - LSTM Model')
plt.savefig('Confusion_Matrix_LSTM.png')

prev_status_test = []
for i in range(len(X_seq_test)):
    seq_length = int(seq_lengths_test[i])
    last_time_step = seq_length - 1
    prev_status = X_seq_test[i, last_time_step, 1]
    prev_status_test.append(prev_status)

prev_status_test = np.array(prev_status_test, dtype=int)
mask = prev_status_test != y_test

X_seq_test_diff = X_seq_test[mask]
X_next_test_diff = X_next_test[mask]
y_test_diff = y_test[mask]

y_pred_probs_diff = model.predict([X_seq_test_diff, X_next_test_diff])
y_pred_classes_diff = np.argmax(y_pred_probs_diff, axis=1)

y_test_diff_labels = status_encoder.inverse_transform(y_test_diff)
y_pred_classes_diff_labels = status_encoder.inverse_transform(y_pred_classes_diff)

conf_matrix_diff = confusion_matrix(
    y_test_diff_labels, y_pred_classes_diff_labels, labels=desired_ordered_labels
)
disp_diff = ConfusionMatrixDisplay(
    confusion_matrix=conf_matrix_diff, display_labels=desired_ordered_labels
)
plt.figure(figsize=(8, 6))
disp_diff.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Changed Statuses')
plt.savefig('Confusion_Matrix_LSTM_Changed_Status.png')

last_utterance_test_entries = [m for m in metadata_test if m['utterance_index'] == 15]

def has_status_change(utterances):
    """
    Check if there is any change in the 'Current Status' within a list of utterances.
    """
    for i in range(1, len(utterances)):
        if utterances[i]['Current Status'] != utterances[i-1]['Current Status']:
            return True
    return False

last_utterance_test_entries = [e for e in last_utterance_test_entries if has_status_change(e['input_data'])]

# Write these entries to a txt file
with open('true_testing_data.txt', 'w') as f:
    for entry in last_utterance_test_entries:
        # You can format this however you'd like. Here we provide all details.
        f.write("Trial: {}\n".format(entry['trial']))
        f.write("Quadrant: {}\n".format(entry['quadrant']))
        f.write("Object: {}\n".format(entry['object']))
        f.write("Input_Data:\n")
        for ut in entry['input_data']:
            f.write("  {}\n".format(ut))
        f.write("--------------------------------------------------\n")
