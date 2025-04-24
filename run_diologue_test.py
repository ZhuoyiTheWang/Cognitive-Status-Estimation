from lstm_predictor import LSTMStatusPredictor

model_path = ".\\trained_models\LSTM\seq_current_status-mentioned-gesture-grammar_role__next_mentioned-gesture-grammar_role\\best_model.keras"
encoders_path = ".\\trained_models\encoders.pkl"
# selected_seq_features = ["current_status", "mentioned", "gesture", "grammar_role"]
# selected_next_features = [""]
predictor = LSTMStatusPredictor(model_path, encoders_path)
'''
Single Utterance Format
    {Uttterance: N} - N is 1-indexed utt number (int)
    {Current Status: C} - N is in {IF, A, F, UI} (str)
    {Mentioned: S} - S is in {M, N} (str)
    {Gesture: G} - is in {TODO: gesture stats} (str)
    {Grammatical Role: R} is in {TODO: gramatical roles}

Sample - list of up-to 15 utterances

Data - list of samples
'''
# data is list of samples
# input_data is list of up to 15 utterance
input_data = []
Utterance1 = {
    "Current Status": 'F',
    "Mentioned": 'M',
    "Gesture": 'N',
    "Grammatical Role": 'NI'
}

input_data.append(Utterance1)
Utterance2 = {
    "Current Status": 'A',
    "Mentioned": 'N',
    "Gesture": 'N',
    "Grammatical Role": 'NI'
}
input_data.append(Utterance2)

next_features = {
    "Mentioned": 'N',
    "Gesture": 'N',
    "Grammatical Role": 'NI'
}

sample1 = {
    "input_data": input_data,
    "next_features": next_features
}

samples = [sample1]

next_cs, prob = predictor.predict(samples)

print(next_cs)
print(prob)



