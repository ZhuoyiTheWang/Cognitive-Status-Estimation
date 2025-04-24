import ast
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

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
                # Possibly an utterance dict
                if reading_utterances and line.startswith('{'):
                    dict_obj = ast.literal_eval(line)
                    utterances.append(dict_obj)

    # last block if file doesn't end with dashes
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
            print(f"Warning: mismatch utterance counts for block {key}. Skipping.")
            continue

        for i in range(len(gt_utterances)):
            gt_label = gt_utterances[i]['Current Status']
            pred_label = pred_utterances[i]['Current Status']

            # Convert np.str_ to standard Python strings if needed
            gt_label = str(gt_label)
            pred_label = str(pred_label)

            y_true.append(gt_label)
            y_pred.append(pred_label)

    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, desired_labels):
    cm = confusion_matrix(y_true, y_pred, labels=desired_labels)

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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=desired_labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")

    ax.set_title("Confusion Matrix - LSTM Model", pad=20)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.tick_params(axis="both", which="major")

    plt.tight_layout()
    plt.savefig("Confusion_Matrix_LSTM_Predefined_Test.png", dpi=300)
    plt.savefig("Confusion_Matrix_LSTM_Predefined_Test.pdf", dpi=300)

def main():
    # 1) Read the ground truth file
    gt_blocks = parse_input_file('models/LSTM/seq_current_status-mentioned-gesture-grammar_role__next_mentioned-gesture-grammar_role/true_testing_data.txt')
    # 2) Read the predictions file
    pred_blocks = parse_input_file('models/LSTM/seq_current_status-mentioned-gesture-grammar_role__next_mentioned-gesture-grammar_role/predicted_output_only_changed.txt')

    # 3) Build dictionaries keyed by (Trial, Quadrant, Object)
    gt_dict = build_block_dict(gt_blocks)
    pred_dict = build_block_dict(pred_blocks)

    # 4) Collect label pairs
    y_true, y_pred = get_label_pairs(gt_dict, pred_dict)

    # 5) Desired order of labels
    desired_ordered_labels = ['UI', 'F', 'A', 'IF']

    # 6) Plot the confusion matrix
    plot_confusion_matrix(y_true, y_pred, desired_ordered_labels)

if __name__ == "__main__":
    main()
