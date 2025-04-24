import os
import ast
import numpy as np
from sklearn.metrics import classification_report

def parse_input_file(filename):
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

def build_block_dict(blocks_list):
    block_dict = {}
    for b in blocks_list:
        key = (b['trial'], b['quadrant'], b['object_name'])
        block_dict[key] = b['utterances']
    return block_dict

def get_label_pairs(gt_dict, pred_dict):
    y_true = []
    y_pred = []

    for key, gt_utterances in gt_dict.items():
        if key not in pred_dict:
            print(f"Warning: block {key} not found in predictions. Skipping.")
            continue

        pred_utterances = pred_dict[key]
        if len(gt_utterances) != len(pred_utterances):
            print(f"Warning: mismatch in utterance counts for block {key}. Skipping.")
            continue

        for i in range(len(gt_utterances)):
            gt_label = str(gt_utterances[i]['Current Status'])
            pred_label = str(pred_utterances[i]['Current Status'])
            y_true.append(gt_label)
            y_pred.append(pred_label)

    return y_true, y_pred

def save_classification_report(y_true, y_pred, desired_labels, output_path):
    report = classification_report(y_true, y_pred, labels=desired_labels, digits=4, zero_division=0)

    with open(output_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=====================\n")
        f.write(report)

    print(f"Classification metrics saved as {output_path}")

def main():
    base_models_dir = ".\\trained_models\LSTM"

    for exp_folder in os.listdir(base_models_dir):
        exp_path = os.path.join(base_models_dir, exp_folder)
        if not os.path.isdir(exp_path):
            continue

        print(f"Processing experiment folder: {exp_folder}")

        true_testing_file = os.path.join(exp_path, "true_testing_data.txt")
        predicted_file = os.path.join(exp_path, "predicted_output_only_changed.txt")

        if not os.path.exists(true_testing_file) or not os.path.exists(predicted_file):
            print(f"Skipping {exp_folder} (missing required files).")
            continue

        gt_blocks = parse_input_file(true_testing_file)
        pred_blocks = parse_input_file(predicted_file)

        gt_dict = build_block_dict(gt_blocks)
        pred_dict = build_block_dict(pred_blocks)

        y_true, y_pred = get_label_pairs(gt_dict, pred_dict)

        if not y_true or not y_pred:
            print(f"Skipping {exp_folder} (no valid data for evaluation).")
            continue

        desired_ordered_labels = ['UI', 'F', 'A', 'IF']

        classification_report_path = os.path.join(exp_path, "classification_report.txt")
        save_classification_report(y_true, y_pred, desired_ordered_labels, classification_report_path)

if __name__ == "__main__":
    main()
