import ast
import tensorflow as tf
import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np

class LSTMStatusPredictor:
    def __init__(self, model_path: str, encoders_path: str):
        '''
        Input:
            model_path (str) - path to .keras model file
            encoders_path (str) - path to .pkl dictionary of LabelEncoder objects
                encoders_dict = {'status_encoder', 'mentioned_encoder', 'gesture_encoder': gesture_encoder, 'grammar_role_encoder'}
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

    def _prepare_data(self, data):
        '''
        Puts data into format that model can use:
        Inputs:
            data - 

        '''
        X_sequences = []
        X_next_features = []

        for sample in data:
            input_sequence = []
            for utt in sample['input_data']:
                input_sequence.append([
                    self.status_encoder.transform([utt['Current Status']])[0],
                    self.mentioned_encoder.transform([utt['Mentioned']])[0],
                    self.gesture_encoder.transform([utt['Gesture']])[0],
                    self.grammar_role_encoder.transform([utt['Grammatical Role']])[0],
                ])
            X_sequences.append(input_sequence)

            # Encode next_features
            next_feats_encoded = [
                self.mentioned_encoder.transform([sample['next_features']['Mentioned']])[0],
                self.gesture_encoder.transform([sample['next_features']['Gesture']])[0],
                self.grammar_role_encoder.transform([sample['next_features']['Grammatical Role']])[0],
            ]
            X_next_features.append(next_feats_encoded)

        # Pad up to the max length used by your model (e.g., 14)
        X_sequences_padded = pad_sequences(X_sequences, maxlen=14, padding='post', dtype='float32')

        return X_sequences_padded, np.array(X_next_features)

    def predict(self, data):
        X_seq, X_next = self._prepare_data(data)
        y_pred_probs = self.model.predict([X_seq, X_next])
        
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_pred_labels = self.status_encoder.inverse_transform(y_pred_classes)
        return y_pred_labels, y_pred_probs
