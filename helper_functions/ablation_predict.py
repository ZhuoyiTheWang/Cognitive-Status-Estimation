import os
import ast
import pickle
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Concatenate, Input

###############################################
#  Modified LSTMStatusPredictor class to allow for abblation tests
###############################################
class LSTMStatusPredictor:
    def __init__(self, model_path: str, encoders_path: str, selected_seq_features, selected_next_features):
        '''
        Input:
            model_path (str) - path to .keras model file
            encoders_path (str) - path to .pkl dictionary of LabelEncoder objects
                encoders_dict = {'status_encoder', 'mentioned_encoder', 'gesture_encoder': gesture_encoder, 'grammar_role_encoder'}
            selected_seq_features (LIST(str)) - list of sequential features used (str), possible options include:
                "current_status"
                "mentioned"
                "gesture"
                "grammar_role"
            selected_next_features (LIST(str)) - list of next features used (str), possible options include:
                "mentioned"
                "gesture"
                "grammar_role"
        '''
        # Load Keras model
        self.model = tf.keras.models.load_model(model_path)
        # Load encoders
        with open(encoders_path, 'rb') as f:
            encoders_dict = pickle.load(f)
        self.status_encoder = encoders_dict['status_encoder']
        self.mentioned_encoder = encoders_dict['mentioned_encoder']
        self.gesture_encoder = encoders_dict['gesture_encoder']
        self.grammar_role_encoder = encoders_dict['grammar_role_encoder']
        # Save which features were used in this experiment.
        self.selected_seq_features = selected_seq_features  # e.g., ["current_status", "mentioned"]
        self.selected_next_features = selected_next_features  # e.g., ["mentioned"]

    def _prepare_data(self, data):
        """
        Prepares data for prediction. For each sample in data, it builds the input sequence
        using only the features in self.selected_seq_features (mapping to the CSV keys) and similarly
        builds the next_features vector using self.selected_next_features.

        Inputs:
        data - 
        Outputs:

        """
        X_sequences = []
        X_next_features = []
        # Define a mapping from our internal feature names to the keys in the utterance dict.
        key_map = {
            "current_status": "Current Status",
            "mentioned": "Mentioned",
            "gesture": "Gesture",
            "grammar_role": "Grammatical Role"
        }
        for sample in data:
            seq = []
            for utt in sample['input_data']:
                features = []
                for feat in self.selected_seq_features:
                    csv_key = key_map.get(feat)
                    # Get value (if missing, supply a default)
                    if csv_key:
                        val = utt.get(csv_key)
                        if val is None:
                            if feat == "current_status":
                                val = "UI"
                            else:
                                val = "N" if feat != "grammar_role" else ""
                    else:
                        val = None
                    # Encode value according to feature
                    if feat == "current_status":
                        encoded = self.status_encoder.transform([val])[0]
                    elif feat == "mentioned":
                        encoded = self.mentioned_encoder.transform([val])[0]
                    elif feat == "gesture":
                        encoded = self.gesture_encoder.transform([val])[0]
                    elif feat == "grammar_role":
                        encoded = self.grammar_role_encoder.transform([val])[0]
                    else:
                        encoded = 0
                    features.append(encoded)
                seq.append(features)
            X_sequences.append(seq)
            
            # Next features: only include allowed features (note: next features never include current_status)
            next_feats = []
            for feat in self.selected_next_features:
                csv_key = key_map.get(feat)
                if csv_key:
                    val = sample['next_features'].get(csv_key)
                    if val is None:
                        val = "N" if feat != "grammar_role" else ""
                else:
                    val = None
                if feat == "mentioned":
                    encoded = self.mentioned_encoder.transform([val])[0]
                elif feat == "gesture":
                    encoded = self.gesture_encoder.transform([val])[0]
                elif feat == "grammar_role":
                    encoded = self.grammar_role_encoder.transform([val])[0]
                else:
                    encoded = 0
                next_feats.append(encoded)
            X_next_features.append(next_feats)
        # Pad the sequences (using default settings; adjust maxlen if needed)
        X_sequences_padded = pad_sequences(X_sequences, maxlen=14, padding='post', dtype='float32')
        X_next_features = np.array(X_next_features) if len(self.selected_next_features) > 0 else None
        return X_sequences_padded, X_next_features

    def predict(self, data):
        X_seq, X_next = self._prepare_data(data)
        if X_next is not None:
            y_pred_probs = self.model.predict([X_seq, X_next])
        else:
            y_pred_probs = self.model.predict(X_seq)
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_pred_labels = self.status_encoder.inverse_transform(y_pred_classes)
        return y_pred_labels, y_pred_probs


###############################################
# 2. Autoregressive predict function (unchanged)
###############################################
def autoregressive_predict(trial, quadrant, object_name, utterances, predictor):    
    """
    Perform autoregressive prediction for a single block (Trial/Quadrant/Object).
    Returns a new list of utterances with predicted 'Current Status' from utterance #2 onward.
    """
    if not utterances:
        return []

    # Keep the first utterance's status as-is
    predicted_utterances = [dict(utterances[0])]

    # For utterances 2..N, predict based on previously predicted statuses
    for i in range(1, len(utterances)):
        # Build input_data from all previously predicted utterances
        input_data = []
        for j in range(i):
            input_data.append({
                'Utterance': predicted_utterances[j]['Utterance'],
                'Current Status': predicted_utterances[j].get('Current Status', "UI"),
                'Mentioned': predicted_utterances[j].get('Mentioned', "N"),
                'Gesture': predicted_utterances[j].get('Gesture', "N"),
                'Grammatical Role': predicted_utterances[j].get('Grammatical Role', "")
            })
        # For the new utterance (i), we only trust the other features
        next_features = {
            'Mentioned': utterances[i].get('Mentioned', "N"),
            'Gesture': utterances[i].get('Gesture', "N"),
            'Grammatical Role': utterances[i].get('Grammatical Role', "")
        }
        sample_for_predictor = [{
            'input_data': input_data,
            'next_features': next_features
        }]
        predicted_labels, _ = predictor.predict(sample_for_predictor)
        predicted_status = predicted_labels[0]
        new_utt = dict(utterances[i])
        new_utt['Current Status'] = str(predicted_status)  # Overwrite with predicted status
        predicted_utterances.append(new_utt)
    return predicted_utterances


###############################################
# 3. Parse the input file (unchanged)
###############################################
def parse_input_file(filename):
    """
    Reads an input file in the following format:
    
    Trial: P7
    Quadrant: 3
    Object: Q4-C31-Gr-Sc
    Input_Data:
      {'Utterance': 1, 'Current Status': 'UI', ...}
      ...
    --------------------------------------------------
    (Next block)
    
    Returns a list of blocks, each block is a dict:
      {
        'trial': str,
        'quadrant': str,
        'object_name': str,
        'utterances': [list of dicts]
      }
    """
    blocks = []
    current_block = {}
    utterances = []
    reading_utterances = False

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Trial:'):
                if current_block:
                    current_block['utterances'] = utterances
                    blocks.append(current_block)
                current_block = {}
                utterances = []
                reading_utterances = False
                current_block['trial'] = line.split(':', 1)[1].strip()
            elif line.startswith('Quadrant:'):
                current_block['quadrant'] = line.split(':', 1)[1].strip()
            elif line.startswith('Object:'):
                current_block['object_name'] = line.split(':', 1)[1].strip()
            elif line.startswith('Input_Data:'):
                reading_utterances = True
            elif line.startswith('--------------------------------------------------'):
                if current_block:
                    current_block['utterances'] = utterances
                    blocks.append(current_block)
                current_block = {}
                utterances = []
                reading_utterances = False
            else:
                if reading_utterances and line.startswith('{'):
                    dict_obj = ast.literal_eval(line)
                    utterances.append(dict_obj)
    if current_block and utterances:
        current_block['utterances'] = utterances
        blocks.append(current_block)
    return blocks


###############################################
# 4. Write predictions to an output file (unchanged)
###############################################
def write_blocks_to_file(blocks, filename):
    """
    Write blocks to a file in the following format:
    
    Trial: ...
    Quadrant: ...
    Object: ...
    Input_Data:
      { ... }
      { ... }
    --------------------------------------------------
    """
    with open(filename, 'w', encoding='utf-8') as out_file:
        for block in blocks:
            trial = block['trial']
            quadrant = block['quadrant']
            object_name = block['object_name']
            utterances = block['utterances']
            out_file.write(f"Trial: {trial}\n")
            out_file.write(f"Quadrant: {quadrant}\n")
            out_file.write(f"Object: {object_name}\n")
            out_file.write("Input_Data:\n")
            for utt in utterances:
                out_file.write(f"  {utt}\n")
            out_file.write("--------------------------------------------------\n")


###############################################
# 5. Main script: Loop over sub-folders, predict and output
###############################################
def main():
    # Base directory containing experiment sub-folders
    base_models_dir = "models"
    # Path to the shared encoders file
    encoders_path = "encoders.pkl"
    
    # Loop over each sub-folder in the models directory
    for exp_folder in os.listdir(base_models_dir):
        exp_path = os.path.join(base_models_dir, exp_folder)
        if not os.path.isdir(exp_path):
            continue

        # Parse the experiment folder name to determine selected features.
        # Expected naming convention: "seq_<feat1>-<feat2>__next_<featA>-<featB>" or "next_none"
        parts = exp_folder.split("__")
        seq_part = parts[0]  # e.g., "seq_current_status-gesture"
        next_part = parts[1] if len(parts) > 1 else "next_none"
        selected_seq_features = seq_part.replace("seq_", "").split("-")
        next_feats_str = next_part.replace("next_", "")
        if next_feats_str.lower() == "none":
            selected_next_features = []
        else:
            selected_next_features = next_feats_str.split("-")
        print(f"Processing experiment folder: {exp_folder}")
        print(f"  Selected sequence features: {selected_seq_features}")
        print(f"  Selected next features: {selected_next_features}")

        # Paths for the model and true_testing_data file in this experiment folder.
        model_path = os.path.join(exp_path, "best_model.keras")
        true_testing_file = os.path.join(exp_path, "true_testing_data.txt")
        
        # Parse the input blocks from the true_testing_data file.
        blocks = parse_input_file(true_testing_file)
        if not blocks:
            print(f"  No blocks found in {true_testing_file}. Skipping.")
            continue

        # Instantiate the predictor with the proper selected features.
        predictor = LSTMStatusPredictor(model_path, encoders_path, selected_seq_features, selected_next_features)
        
        # For each block, run autoregressive prediction.
        output_blocks = []
        for block in blocks:
            trial = block.get('trial')
            quadrant = block.get('quadrant')
            object_name = block.get('object_name')
            utterances = block.get('utterances', [])
            if not utterances:
                output_blocks.append(block)
                continue

            predicted_utterances = autoregressive_predict(trial, quadrant, object_name, utterances, predictor)
            
            new_block = {
                'trial': trial,
                'quadrant': quadrant,
                'object_name': object_name,
                'utterances': predicted_utterances
            }
            
            output_blocks.append(new_block)
        # Write the predictions to an output file in the experiment folder.
        output_file = os.path.join(exp_path, "predicted_output_only_changed.txt")
        write_blocks_to_file(output_blocks, output_file)
        print(f"Predictions written to {output_file}\n")

if __name__ == "__main__":
    main()
