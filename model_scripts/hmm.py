import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm

# =============================================================================
# 1. Parse Predefined Test Entries from File
# =============================================================================
test_combinations = []
with open("true_testing_data.txt", "r") as f:
    current_entry = {}
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Trial:"):
            current_entry["trial"] = line.split("Trial:")[1].strip()
        elif line.startswith("Quadrant:"):
            current_entry["quadrant"] = line.split("Quadrant:")[1].strip()
        elif line.startswith("Object:"):
            current_entry["object"] = line.split("Object:")[1].strip()
        elif line.startswith("--------------------------------------------------"):
            if "trial" in current_entry and "quadrant" in current_entry and "object" in current_entry:
                test_combinations.append((current_entry["trial"], current_entry["quadrant"], current_entry["object"]))
            current_entry = {}
    if current_entry and "trial" in current_entry and "quadrant" in current_entry and "object" in current_entry:
        test_combinations.append((current_entry["trial"], current_entry["quadrant"], current_entry["object"]))
print("Predefined test combinations:", test_combinations)

# =============================================================================
# 2. Load and Process Master Dataset
# =============================================================================
samples = []
directory = Path("data/Master Dataset")
all_files = [f for f in directory.rglob("*") if f.is_file()]

for file in all_files:
    trial = file.stem
    df = pd.read_csv(file, dtype=str).drop_duplicates()
    groups = df.groupby(["Quadrant", "Object"])
    for (quadrant, obj), group in groups:
        group["Utterance"] = group["Utterance"].astype(int)
        group = group.sort_values("Utterance").drop_duplicates(subset="Utterance", keep="first").reset_index(drop=True)
        for i in range(len(group)):
            if i == 0:
                context = {
                    "Previous Status": "F" if obj.split("-")[0] == ("Q" + str(quadrant)) else "UI",
                    "Mentioned": group.iloc[i]["Mentioned"],
                    "Gesture": group.iloc[i]["Gesture"],
                    "Grammatical Role": group.iloc[i]["Grammatical Role"]
                }
            else:
                context = {
                    "Previous Status": group.iloc[i - 1]["Current Status"],
                    "Mentioned": group.iloc[i]["Mentioned"],
                    "Gesture": group.iloc[i]["Gesture"],
                    "Grammatical Role": group.iloc[i]["Grammatical Role"]
                }
            target = group.iloc[i]["Current Status"]
            samples.append({
                "trial": trial,
                "quadrant": str(quadrant),
                "object": obj,
                "feature": context,
                "target": target
            })
print("Total samples generated:", len(samples))

# =============================================================================
# 3. Encode Features and Targets
# =============================================================================
status_encoder = LabelEncoder()
mentioned_encoder = LabelEncoder()
gesture_encoder = LabelEncoder()
grammar_encoder = LabelEncoder()

status_vals = [s["feature"]["Previous Status"] for s in samples] + [s["target"] for s in samples]
mentioned_vals = [s["feature"]["Mentioned"] for s in samples]
gesture_vals = [s["feature"]["Gesture"] for s in samples]
grammar_vals = [s["feature"]["Grammatical Role"] for s in samples]

status_encoder.fit(status_vals)
mentioned_encoder.fit(mentioned_vals)
gesture_encoder.fit(gesture_vals)
grammar_encoder.fit(grammar_vals)

encoders = {
    "status": status_encoder,
    "mentioned": mentioned_encoder,
    "gesture": gesture_encoder,
    "grammar": grammar_encoder
}
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Build observation vector (X) and target vector (y)
X = []
y = []
sample_keys = []
for s in samples:
    feat = s["feature"]
    fv = [
        status_encoder.transform([feat["Previous Status"]])[0],
        mentioned_encoder.transform([feat["Mentioned"]])[0],
        gesture_encoder.transform([feat["Gesture"]])[0],
        grammar_encoder.transform([feat["Grammatical Role"]])[0]
    ]
    X.append(fv)
    y.append(status_encoder.transform([s["target"]])[0])
    sample_keys.append((s["trial"], s["quadrant"], s["object"]))
X = np.array(X)
y = np.array(y)

# =============================================================================
# 4. Split Data into Predefined Test Set and Train/Validation Set
# =============================================================================
test_idx = [i for i, key in enumerate(sample_keys) if key in test_combinations]
train_val_idx = [i for i, key in enumerate(sample_keys) if key not in test_combinations]

X_test = X[test_idx]
y_test = y[test_idx]
X_train = X[train_val_idx]
y_train = y[train_val_idx]

# =============================================================================
# 5. Create Composite Observations for CategoricalHMM
# =============================================================================
composite_dict = {}
X_composite = []
for obs in X:
    t = tuple(obs)
    if t not in composite_dict:
        composite_dict[t] = len(composite_dict)
    X_composite.append(composite_dict[t])
X_composite = np.array(X_composite).reshape(-1, 1)

# Update training and test splits for composite observations:
X_train_comp = X_composite[train_val_idx]
X_test_comp = X_composite[test_idx]

# =============================================================================
# 6. Compute Transition Probabilities from y (targets)
# =============================================================================
n_states = len(status_encoder.classes_)
grouped_samples_train = {}
for i in train_val_idx:
    key = sample_keys[i]
    grouped_samples_train.setdefault(key, []).append(i)

transition_counts = np.zeros((n_states, n_states))
for key, indices in grouped_samples_train.items():
    indices.sort()
    for j in range(len(indices) - 1):
        prev_idx = indices[j]
        curr_idx = indices[j + 1]
        prev_state = y[prev_idx]
        curr_state = y[curr_idx]
        transition_counts[prev_state, curr_state] += 1

transition_probs = transition_counts / transition_counts.sum(axis=1, keepdims=True)
transition_probs[np.isnan(transition_probs)] = 1.0 / n_states

# Define the desired order for states.
desired_ordered_labels = ["UI", "F", "A", "IF"]
# Map from alphabetical order to desired order.
reorder = [list(status_encoder.classes_).index(label) for label in desired_ordered_labels]

print("Transition Probability Matrix in order UI, F, A, IF:")
print(transition_probs[reorder, :][:, reorder])

train_lengths = [len(indices) for indices in grouped_samples_train.values()]

# =============================================================================
# 7A. Manually Compute and Smooth Emission Probabilities
# =============================================================================
# We'll count how many times each state emits each composite symbol in the training set.
emission_counts = np.zeros((n_states, len(composite_dict)), dtype=float)

for i in range(len(X_train_comp)):
    obs_id = X_train_comp[i][0]  # composite symbol index
    st = y_train[i]              # state index
    emission_counts[st, obs_id] += 1.0

# Add Laplace smoothing: a small epsilon to avoid zeros.
epsilon = 1e-3
emission_counts += epsilon

# Normalize each row so that the probabilities for each state sum to 1.
emission_probs = emission_counts / emission_counts.sum(axis=1, keepdims=True)

# =============================================================================
# 7B. Create and Initialize the HMM with Fixed Parameters
# =============================================================================
# We set init_params="" so hmmlearn won't overwrite startprob_ or transmat_.
# We set params="" so it won't re-estimate them either. 
# We'll set n_iter=1 or 0 (makes no difference here, as no EM steps will be performed).
n_features = len(composite_dict)
model = hmm.CategoricalHMM(
    n_components=n_states,
    n_iter=1,
    n_features=n_features,
    init_params="",
    params=""   # No parameters are re-estimated
)

# Compute the empirical start distribution from training sequences:
start_counts = np.zeros(n_states)
# Assuming grouped_samples_train is a dict: key -> list of indices (global) in training set
for indices in grouped_samples_train.values():
    first_state = y[indices[0]]
    start_counts[first_state] += 1

empirical_start_prob = start_counts / start_counts.sum()
print("Empirical start distribution:", empirical_start_prob)
model.startprob_ = empirical_start_prob

# Transition probabilities from the code
model.transmat_ = transition_probs
# Emission probabilities from manual smoothing
model.emissionprob_ = emission_probs

# =============================================================================
# 8. Predict States on Test Composite Observations (for evaluation)
# =============================================================================
y_pred_hmm = model.predict(X_test_comp)
print("X_test_comp shape:", X_test_comp.shape)

# Convert numeric predictions to labels for confusion matrix
y_test_labels_hmm = status_encoder.inverse_transform(y_test)
y_pred_labels_hmm = status_encoder.inverse_transform(y_pred_hmm)

cm_hmm = confusion_matrix(y_test_labels_hmm, y_pred_labels_hmm, labels=desired_ordered_labels)
sns.set_theme(style="white")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14
})
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_hmm, display_labels=desired_ordered_labels)
disp.plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Confusion Matrix - Categorical HMM Model", pad=20)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
plt.tight_layout()
plt.savefig("Confusion_Matrix_CategoricalHMM_Smoothed.png", dpi=300)
plt.savefig("Confusion_Matrix_CategoricalHMM_Smoothed.pdf", dpi=300)
plt.show()

# =============================================================================
# 9. Optional: Run Forward Algorithm for All Test Sequences and Write Details to a File
# =============================================================================
grouped_samples_test = {}
for local_idx, global_idx in enumerate(test_idx):
    key = sample_keys[global_idx]
    grouped_samples_test.setdefault(key, []).append(local_idx)

def forward_algorithm(obs_seq, startprob, transmat, emissionprob, file_handle, reorder):
    """
    Compute the forward probabilities for a sequence of composite observations.
    For each time step, write to file the emission probability for the observation,
    the contributions from previous states (via the transition probabilities),
    and the resulting normalized state probabilities. The output is ordered 
    according to the 'reorder' list (UI, F, A, IF).
    """
    n_states = len(startprob)
    T = len(obs_seq)
    alpha = np.zeros((T, n_states))
    
    # Time 0
    obs0 = int(obs_seq[0][0])
    alpha[0] = startprob * emissionprob[:, obs0]
    file_handle.write(f"\nTime 0: Observation index = {obs0}\n")
    startprob_reordered = np.array(startprob)[reorder]
    file_handle.write("Start probabilities (UI, F, A, IF): " + str(startprob_reordered) + "\n")
    emis0 = emissionprob[:, obs0][reorder]
    file_handle.write("Emission probabilities (UI, F, A, IF): " + str(emis0) + "\n")
    file_handle.write("Alpha[0] (unnormalized, reordered): " + str(alpha[0][reorder]) + "\n")
    alpha[0] = alpha[0] / np.sum(alpha[0])
    file_handle.write("Alpha[0] (normalized, reordered): " + str(alpha[0][reorder]) + "\n")
    
    for t in range(1, T):
        obs_t = int(obs_seq[t][0])
        file_handle.write(f"\nTime {t}: Observation index = {obs_t}\n")
        for j in reorder:
            contrib = alpha[t-1] * transmat[:, j]
            sum_contrib = np.sum(contrib)
            unnorm = emissionprob[j, obs_t] * sum_contrib
            alpha[t, j] = unnorm
            state_label = status_encoder.inverse_transform([j])[0]
            file_handle.write(f"  State {state_label}:\n")
            file_handle.write(f"    Emission probability: {emissionprob[j, obs_t]}\n")
            file_handle.write(f"    Contributions (alpha[t-1] * transmat[:, {j}]): {contrib}\n")
            file_handle.write(f"    Sum of contributions: {sum_contrib}\n")
            file_handle.write(f"    Unnormalized alpha[t, {state_label}]: {unnorm}\n")
        norm_factor = np.sum(alpha[t])
        if norm_factor > 0:
            alpha[t] = alpha[t] / norm_factor
        file_handle.write("Normalized state probabilities (UI, F, A, IF): " + str(alpha[t][reorder]) + "\n")
    return alpha

with open("forward_algorithm_all_tests_smoothed.txt", "w") as out_file:
    out_file.write("Forward Algorithm Details for All Test Sequences (Smoothed)\n")
    out_file.write("==========================================================\n")
    for key, local_indices in grouped_samples_test.items():
        local_indices.sort()
        test_seq = X_test_comp[local_indices]  # Local indices in X_test_comp
        out_file.write(f"\n\nTest Sequence with key: {key}\n")
        out_file.write("-----------------------------------------------\n")
        alpha = forward_algorithm(
            test_seq, model.startprob_, model.transmat_, model.emissionprob_, 
            out_file, reorder
        )

print("\nForward algorithm details (with smoothing) have been saved to 'forward_algorithm_all_tests_smoothed.txt'.")