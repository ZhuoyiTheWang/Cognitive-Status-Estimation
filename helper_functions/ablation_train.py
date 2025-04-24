import os
import pandas as pd
import numpy as np
import itertools
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Concatenate, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# =============================================================================
# 1. LOAD THE DATA (as provided)
# =============================================================================

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
                    'quadrant': object_name,
                    'object': object_name,
                    'utterance_index': i+1,
                    'input_data': input_data,
                })

# =============================================================================
# 2. SET UP LABEL ENCODERS (for each feature)
# =============================================================================
with open('encoders.pkl', 'rb') as f:
    encoders_dict = pickle.load(f)
status_encoder = encoders_dict['status_encoder']
mentioned_encoder = encoders_dict['mentioned_encoder']
gesture_encoder = encoders_dict['gesture_encoder']
grammar_role_encoder = encoders_dict['grammar_role_encoder']

# =============================================================================
# 3. HELPER FUNCTIONS FOR EXPERIMENTS
# =============================================================================

def prepare_data(selected_seq_features, selected_next_features):
    """
    Prepare the padded sequence inputs, next_features vectors, targets, and sequence lengths.
    
    For each sample, the full sequence vector (per utterance) is:
      [Utterance, current_status, mentioned, gesture, grammar_role]
    We ignore the utterance number and pick only those features specified in selected_seq_features.
    
    For each utterance in a sample, the mapping is:
      - "current_status" -> use status_encoder on 'Current Status'
      - "mentioned"      -> use mentioned_encoder on 'Mentioned'
      - "gesture"        -> use gesture_encoder on 'Gesture'
      - "grammar_role"   -> use grammar_role_encoder on 'Grammatical Role'
    
    For the next features, we pick (in order) the ones in selected_next_features from the full next_features:
      The full next_features (for each sample) come from:
         ['Mentioned', 'Gesture', 'Grammatical Role']
    """
    # Mapping for sequence features (ignoring the Utterance index)
    seq_map = {
        "current_status": lambda ut: status_encoder.transform([ut['Current Status']])[0],
        "mentioned": lambda ut: mentioned_encoder.transform([ut['Mentioned']])[0],
        "gesture": lambda ut: gesture_encoder.transform([ut['Gesture']])[0],
        "grammar_role": lambda ut: grammar_role_encoder.transform([ut['Grammatical Role']])[0]
    }
    
    X_sequences = []
    X_next_features = []
    y = []
    sequence_lengths = []
    
    for sample in all_training_samples:
        full_seq = []
        for utterance in sample['input_data']:
            # Build the vector for this utterance
            vec = {
                "current_status": seq_map["current_status"](utterance),
                "mentioned": seq_map["mentioned"](utterance),
                "gesture": seq_map["gesture"](utterance),
                "grammar_role": seq_map["grammar_role"](utterance)
            }
            full_seq.append(vec)
        # For the sequence input, pick only the desired features (in the order given by selected_seq_features)
        seq = [[vec[feat] for feat in selected_seq_features] for vec in full_seq]
        sequence_lengths.append(len(seq))
        
        # For next features, note that we want to use the same features as in the seq input (if valid)
        # But next features never include "current_status"
        next_feats = []
        for feat in selected_next_features:
            # Use the sample's next_features dictionary – keys are capitalized in the CSV
            if feat == "mentioned":
                val = mentioned_encoder.transform([sample['next_features']['Mentioned']])[0]
            elif feat == "gesture":
                val = gesture_encoder.transform([sample['next_features']['Gesture']])[0]
            elif feat == "grammar_role":
                val = grammar_role_encoder.transform([sample['next_features']['Grammatical Role']])[0]
            next_feats.append(val)
        
        # The target is always the current status of the current utterance.
        target = status_encoder.transform([sample['target']])[0]
        
        X_sequences.append(seq)
        X_next_features.append(next_feats)
        y.append(target)
    
    X_sequences_padded = pad_sequences(X_sequences, padding='post', dtype='float32')
    # If no next features were selected, we return None (and later pass only one model input)
    X_next_features = np.array(X_next_features) if len(selected_next_features) > 0 else None
    y = np.array(y)
    sequence_lengths = np.array(sequence_lengths)
    
    return X_sequences_padded, X_next_features, y, sequence_lengths


def build_model(seq_input_shape, next_features_shape, num_classes):
    """
    Build and compile an LSTM-based model.
    
    If next_features_shape is None (or 0), the model will have only the sequence input.
    Otherwise, it will take two inputs and concatenate the outputs.
    """
    sequence_input = Input(shape=(seq_input_shape[0], seq_input_shape[1]), name="sequence_input")
    x = LSTM(128, return_sequences=True)(sequence_input)
    x = Dropout(0.1)(x)
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.1)(x)
    x = LSTM(64)(x)
    x = Dropout(0.1)(x)
    
    if next_features_shape is not None and next_features_shape[0] > 0:
        next_features_input = Input(shape=(next_features_shape[0],), name="next_features_input")
        concat = Concatenate()([x, next_features_input])
        inputs = [sequence_input, next_features_input]
    else:
        concat = x
        inputs = sequence_input  # single-input model
    
    fc = Dense(64, activation='relu')(concat)
    fc = Dropout(0.1)(fc)
    fc = Dense(32, activation='relu')(fc)
    fc = Dropout(0.1)(fc)
    output = Dense(num_classes, activation='softmax')(fc)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def has_status_change(utterances):
    """
    Check if there is any change in the 'Current Status' within a list of utterances.
    """
    for i in range(1, len(utterances)):
        if utterances[i]['Current Status'] != utterances[i-1]['Current Status']:
            return True
    return False


def write_true_testing_data(metadata_test, output_filepath, selected_seq_features):
    """
    From the metadata_test list, filter the entries corresponding to the last utterance (assumed to be utterance_index==15)
    and then further filter to keep only those entries where a status change occurred.
    
    For each entry, only include in the output the keys used in this experiment. That is, always include 'Utterance'
    and the output keys corresponding to the selected sequence features.
    
    The mapping from our internal feature names to the CSV keys is:
      - "current_status" -> "Current Status"
      - "mentioned"      -> "Mentioned"
      - "gesture"        -> "Gesture"
      - "grammar_role"   -> "Grammatical Role"
    """
    key_map = {
        "mentioned": "Mentioned",
        "gesture": "Gesture",
        "grammar_role": "Grammatical Role"
    }

    # Ensure all last-utterance entries with a status change are included.
    last_utterance_test_entries = [m for m in metadata_test if m['utterance_index'] == 15]
    last_utterance_test_entries = [e for e in last_utterance_test_entries if has_status_change(e['input_data'])]
    
    with open(output_filepath, 'w') as f:
        for entry in last_utterance_test_entries:
            f.write(f"Trial: {entry['trial']}\n")
            f.write(f"Quadrant: {entry['quadrant']}\n")
            f.write(f"Object: {entry['object']}\n")
            f.write("Input_Data:\n")
            
            for ut in entry['input_data']:
                new_ut = {"Utterance": ut["Utterance"], "Current Status": ut["Current Status"]}  # Always include Current Status
                for feat in selected_seq_features:
                    if feat != "current_status":  # Avoid duplicating Current Status
                        csv_key = key_map.get(feat)
                        if csv_key and csv_key in ut:
                            new_ut[csv_key] = ut[csv_key]
                f.write(f"  {new_ut}\n")
            f.write("--------------------------------------------------\n")
    print(f"Saved true testing data to {output_filepath}")


def get_experiment_id(seq_feats, next_feats):
    """Generate a unique string identifier from the feature lists."""
    seq_str = "seq_" + "-".join(seq_feats)
    next_str = "next_" + ("-".join(next_feats) if next_feats else "none")
    return f"{seq_str}__{next_str}"


# =============================================================================
# 4. EXPERIMENT LOOP
# =============================================================================

# Define the available features for the sequence input.
available_features = ["current_status", "mentioned", "gesture", "grammar_role"]

# We will loop over all non-empty subsets of these features.
seq_feature_subsets = []
for r in range(1, len(available_features) + 1):
    for subset in itertools.combinations(available_features, r):
        seq_feature_subsets.append(list(subset))

# For consistency, precompute a train/test split using the full data (all features).
X_seq_full, X_next_full, y_full, seq_lengths_full = prepare_data(
    available_features,
    [feat for feat in ["mentioned", "gesture", "grammar_role"] if feat in available_features]
)
indices = np.arange(len(y_full))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

# Loop over every experiment.
for seq_feats in seq_feature_subsets:
    next_feats = [feat for feat in ["mentioned", "gesture", "grammar_role"] if feat in seq_feats]
    
    exp_id = get_experiment_id(seq_feats, next_feats)
    exp_dir = os.path.join("models", exp_id)
    
    # Check if the experiment directory exists; if so, skip training
    if os.path.exists(exp_dir):
        print(f"\nSkipping experiment '{exp_id}' as it already exists.")
        continue
    
    print("\n==============================")
    print("Running experiment:", exp_id)
    print("Sequence features:", seq_feats)
    print("Next features    :", next_feats if next_feats else "None")
    print("==============================")

    # Create the directory for the experiment
    os.makedirs(exp_dir, exist_ok=True)
    
    # Prepare data for this experiment
    X_seq, X_next, y, seq_lengths = prepare_data(seq_feats, next_feats)

    # Use the precomputed train/test indices.
    X_seq_train = X_seq[train_idx]
    X_seq_test = X_seq[test_idx]
    if X_next is not None:
        X_next_train = X_next[train_idx]
        X_next_test = X_next[test_idx]
    else:
        X_next_train, X_next_test = None, None
    y_train = y[train_idx]
    y_test = y[test_idx]
    metadata_arr = np.array(metadata)
    metadata_train = metadata_arr[train_idx]
    metadata_test = metadata_arr[test_idx]
    
    # Build the model
    if X_next_train is not None:
        next_input_shape = (X_next_train.shape[1],)
    else:
        next_input_shape = (0,)

    model = build_model(
        seq_input_shape=X_seq_train.shape[1:],  
        next_features_shape=next_input_shape,
        num_classes=len(status_encoder.classes_)
    )

    # Create callbacks with model saved in the experiment folder
    model_filepath = os.path.join(exp_dir, "best_model.keras")
    model_checkpoint_callback = ModelCheckpoint(
        filepath=model_filepath,
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

    # Train the model
    if X_next_train is not None:
        history = model.fit(
            [X_seq_train, X_next_train],
            y_train,
            epochs=200,
            batch_size=16,
            validation_split=0.1,
            verbose=1,
            callbacks=[model_checkpoint_callback, early_stopping_callback]
        )
    else:
        history = model.fit(
            X_seq_train,
            y_train,
            epochs=200,
            batch_size=16,
            validation_split=0.1,
            verbose=1,
            callbacks=[model_checkpoint_callback, early_stopping_callback]
        )

    # Evaluate on the test set
    if X_next_test is not None:
        test_loss, test_accuracy = model.evaluate([X_seq_test, X_next_test], y_test, verbose=1)
    else:
        test_loss, test_accuracy = model.evaluate(X_seq_test, y_test, verbose=1)
    print(f"Experiment '{exp_id}' Test Accuracy: {test_accuracy:.4f}")

    # Write out the "true testing data" file inside the experiment folder
    testing_data_filepath = os.path.join(exp_dir, "true_testing_data.txt")
    write_true_testing_data(metadata_test.tolist(), testing_data_filepath, seq_feats)

