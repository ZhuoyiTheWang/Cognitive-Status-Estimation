import os
import pandas as pd

# Directory containing your experiment sub-folders (each with a classification_report.txt)
models_dir = "models"

# Mapping for converting internal feature names to human-readable names
feature_mapping = {
    "current_status": "Current Status",
    "mentioned": "Mentioned",
    "gesture": "Gesture",
    "grammar_role": "Grammatical Role"
}

# List to hold one dictionary per experiment (each dict becomes one row in the Excel sheet)
experiment_metrics = []

# Define the classes we care about
target_classes = ["UI", "F", "A", "IF"]

# Iterate over each experiment directory
for exp_name in os.listdir(models_dir):
    exp_path = os.path.join(models_dir, exp_name)
    if os.path.isdir(exp_path):
        # Parse the experiment folder name to a more readable format.
        # The expected folder name format is:
        # "seq_<feature1>-<feature2>-...__next_<...>"
        # We ignore the next_features part.
        if '__' in exp_name:
            seq_part = exp_name.split('__')[0]
            if seq_part.startswith("seq_"):
                seq_features_str = seq_part[4:]  # remove the "seq_" prefix
            else:
                seq_features_str = seq_part
            # Split the sequence features by '-' and convert them using the mapping.
            features_list = seq_features_str.split('-')
            readable_features = [feature_mapping.get(feat, feat) for feat in features_list]
            readable_exp_name = ", ".join(readable_features)
        else:
            # If the folder name does not follow the expected format, keep it as is.
            readable_exp_name = exp_name
        
        report_file = os.path.join(exp_path, "classification_report.txt")
        if os.path.exists(report_file):
            with open(report_file, "r") as f:
                lines = f.readlines()
            
            # Initialize dictionary for this experiment with all the desired metric fields.
            metrics = {
                "Experiment": readable_exp_name,
                "Accuracy": None,
                "UI Precision": None,
                "UI Recall": None,
                "UI F1-Score": None,
                "F Precision": None,
                "F Recall": None,
                "F F1-Score": None,
                "A Precision": None,
                "A Recall": None,
                "A F1-Score": None,
                "IF Precision": None,
                "IF Recall": None,
                "IF F1-Score": None,
            }
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                
                # Check for the overall accuracy line.
                # Example format:
                # "accuracy                         0.8848       330"
                if parts[0] == "accuracy":
                    try:
                        metrics["Accuracy"] = float(parts[1])
                    except Exception as e:
                        print(f"Error parsing accuracy in experiment '{exp_name}': {e}")
                    continue

                # For class rows, we expect 5 tokens: label, precision, recall, f1-score, support.
                if len(parts) == 5:
                    label = parts[0]
                    if label in target_classes:
                        try:
                            precision = float(parts[1])
                            recall    = float(parts[2])
                            f1_score  = float(parts[3])
                        except Exception as e:
                            print(f"Error parsing metrics for {label} in experiment '{exp_name}': {e}")
                            continue
                        metrics[f"{label} Precision"] = precision
                        metrics[f"{label} Recall"]    = recall
                        metrics[f"{label} F1-Score"]   = f1_score
            
            experiment_metrics.append(metrics)
        else:
            print(f"Warning: {report_file} not found in experiment '{exp_name}'. Skipping.")

# Create a DataFrame from the collected metrics and save it as an Excel file.
df = pd.DataFrame(experiment_metrics)
excel_filename = "experiment_metrics.xlsx"
df.to_excel(excel_filename, index=False)
print(f"Metrics saved to {excel_filename}")