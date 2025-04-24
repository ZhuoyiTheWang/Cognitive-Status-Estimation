import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. Parse Predefined Test Entries from File
# =============================================================================
# The file "true_testing_data.txt" lists entries (Trial, Quadrant, Object)
# that you want in your test set.
test_combinations = []  # List of tuples: (trial, quadrant, object)
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
# 2. Generate Samples from the Master Dataset
# =============================================================================
# For each CSV, we group by Quadrant and Object and generate one sample per utterance.
# Each sample uses only the previous (context) information:
#  - For the first utterance, we use default values.
#  - For subsequent utterances, we use the previous row’s values.
samples = []  # Each sample is a dict with keys: trial, quadrant, object, feature, target
directory = Path("data/Master Dataset")
all_files = [f for f in directory.rglob("*") if f.is_file()]

# Default context info for the first utterance in a group.
def default_context(object_name, quadrant):
    return {
        "Current Status": "F" if object_name.split("-")[0] == ("Q" + str(quadrant)) else "UI",
        "Mentioned": "N",
        "Gesture": "N",
        "Grammatical Role": ""
    }

for file in all_files:
    trial = file.stem
    df = pd.read_csv(file, dtype=str)
    df = df.drop_duplicates()
    # Group by Quadrant and Object
    groups = df.groupby(["Quadrant", "Object"])
    for (quadrant, obj), group in groups:
        group["Utterance"] = group["Utterance"].astype(int)
        group = group.sort_values("Utterance").drop_duplicates(subset="Utterance", keep="first")
        group = group.reset_index(drop=True)
        for i in range(len(group)):
            # Determine the context: for i==0, use default; otherwise, use previous row's info.
            if i == 0:
                context = default_context(obj, quadrant)
            else:
                prev_row = group.iloc[i-1]
                context = {
                    "Current Status": prev_row["Current Status"],
                    "Mentioned": prev_row["Mentioned"],
                    "Gesture": prev_row["Gesture"],
                    "Grammatical Role": prev_row["Grammatical Role"]
                }
            target = group.iloc[i]["Current Status"]
            sample = {
                "trial": trial,
                "quadrant": str(quadrant),
                "object": obj,
                "feature": context,  # This is our flat feature dict
                "target": target
            }
            samples.append(sample)

print("Total samples generated:", len(samples))

# =============================================================================
# 3. Encode Features and Targets
# =============================================================================
# We need to encode:
#   - The context's "Current Status" (as a feature) using status_encoder.
#   - "Mentioned", "Gesture", and "Grammatical Role" using their respective encoders.
#   - The target (also a status) using status_encoder.
status_encoder = LabelEncoder()
mentioned_encoder = LabelEncoder()
gesture_encoder = LabelEncoder()
grammar_encoder = LabelEncoder()

# Build lists for fitting encoders
status_vals = [s["feature"]["Current Status"] for s in samples] + [s["target"] for s in samples]
mentioned_vals = [s["feature"]["Mentioned"] for s in samples]
gesture_vals = [s["feature"]["Gesture"] for s in samples]
grammar_vals = [s["feature"]["Grammatical Role"] for s in samples]

status_encoder.fit(status_vals)
mentioned_encoder.fit(mentioned_vals)
gesture_encoder.fit(gesture_vals)
grammar_encoder.fit(grammar_vals)

# Optionally, save encoders
encoders = {
    "status": status_encoder,
    "mentioned": mentioned_encoder,
    "gesture": gesture_encoder,
    "grammar": grammar_encoder
}
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# Transform samples into feature and target arrays.
# The FNN feature vector is: [encoded(Previous Status), encoded(Mentioned), encoded(Gesture), encoded(Grammatical Role)]
X = []
y = []
sample_keys = []  # We'll store (trial, quadrant, object) for each sample.
for s in samples:
    feat = s["feature"]
    fv = [
        status_encoder.transform([feat["Current Status"]])[0],
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
# A sample is in the test set if its key (trial, quadrant, object) is in test_combinations.
test_idx = [i for i, key in enumerate(sample_keys) if key in test_combinations]
train_val_idx = [i for i, key in enumerate(sample_keys) if key not in test_combinations]

X_test = X[test_idx]
y_test = y[test_idx]
X_train = X[train_val_idx]
y_train= y[train_val_idx]


# =============================================================================
# 5. Build and Train the FNN Model
# =============================================================================
num_classes = len(status_encoder.classes_)
checkpoint_path = "best_fn_model.keras"
model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

model = Sequential([
    Dense(16, activation="relu", input_shape=(X.shape[1],)),
    Dense(8, activation="relu"),
    Dense(8, activation="relu"),
    Dense(8, activation="relu"),
    Dense(num_classes, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=50, validation_split=0.1,
                    batch_size=16, verbose=1, callbacks=[model_checkpoint_callback])
model = tf.keras.models.load_model(checkpoint_path)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"FNN Test Accuracy: {test_accuracy:.4f}")

# =============================================================================
# 6. Generate Predictions and Plot the Confusion Matrix for the Test Set
# =============================================================================
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)
y_test_labels = status_encoder.inverse_transform(y_test)
y_pred_labels = status_encoder.inverse_transform(y_pred_classes)
desired_ordered_labels = ["UI", "F", "A", "IF"]
cm = confusion_matrix(y_test_labels, y_pred_labels, labels=desired_ordered_labels)

sns.set_theme(style="white")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 16,        # General font size
    "axes.titlesize": 20,   # Title font size
    "axes.labelsize": 18,   # Axis label font size
    "xtick.labelsize": 16,  # X-axis tick labels
    "ytick.labelsize": 16,  # Y-axis tick labels
    "legend.fontsize": 14   # Legend font size (if used)
})

fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=desired_ordered_labels)
disp.plot(ax=ax, cmap="Blues", values_format="d")

ax.set_title("Confusion Matrix - FNN Model", pad=20)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.tick_params(axis="both", which="major")

plt.tight_layout()
plt.savefig("Confusion_Matrix_FNN_Predefined_Test.png", dpi=300)
plt.savefig("Confusion_Matrix_FNN_Predefined_Test.pdf", dpi=300)
