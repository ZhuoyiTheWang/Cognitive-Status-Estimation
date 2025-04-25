import ast
import matplotlib.pyplot as plt
import os

##############################################################################
# 1. A function to parse the text file into a dictionary keyed by (trial, quadrant, object_name).
##############################################################################
def parse_input_file(filename):
    """
    Reads the file containing multiple blocks:

      Trial: P7
      Quadrant: 3
      Object: Q4-C31-Gr-Sc
      Input_Data:
        {'Utterance': 1, 'Current Status': 'UI', ...}
        ...
      --------------------------------------------------
      (Next block)
      ...
    
    Returns a dictionary:
      {
        (trial, quadrant, object_name): {
          'trial': str,
          'quadrant': str,
          'object_name': str,
          'utterances': [list of dicts sorted by 'Utterance']
        },
        ...
      }
    """
    blocks_dict = {}
    current_block = {}
    utterances = []
    reading_utterances = False

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Trial:'):
                # If there's a previous block, store it in blocks_dict
                if current_block and 'trial' in current_block and 'quadrant' in current_block and 'object_name' in current_block:
                    # Sort the utterances by 'Utterance' number
                    utterances = sorted(utterances, key=lambda x: x['Utterance'])
                    key = (current_block['trial'], current_block['quadrant'], current_block['object_name'])
                    blocks_dict[key] = {
                        'trial': current_block['trial'],
                        'quadrant': current_block['quadrant'],
                        'object_name': current_block['object_name'],
                        'utterances': utterances
                    }
                # Start a new block
                current_block = {}
                utterances = []
                reading_utterances = False
                # Extract the trial value
                current_block['trial'] = line.split(':', 1)[1].strip()

            elif line.startswith('Quadrant:'):
                current_block['quadrant'] = line.split(':', 1)[1].strip()

            elif line.startswith('Object:'):
                current_block['object_name'] = line.split(':', 1)[1].strip()

            elif line.startswith('Input_Data:'):
                reading_utterances = True

            elif line.startswith('--------------------------------------------------'):
                # End of a block
                if current_block and 'trial' in current_block and 'quadrant' in current_block and 'object_name' in current_block:
                    # Sort the utterances by 'Utterance'
                    utterances = sorted(utterances, key=lambda x: x['Utterance'])
                    key = (current_block['trial'], current_block['quadrant'], current_block['object_name'])
                    blocks_dict[key] = {
                        'trial': current_block['trial'],
                        'quadrant': current_block['quadrant'],
                        'object_name': current_block['object_name'],
                        'utterances': utterances
                    }
                current_block = {}
                utterances = []
                reading_utterances = False

            else:
                # Possibly an utterance line
                if reading_utterances and line.startswith('{'):
                    # Safely parse the dict-like line
                    dict_obj = ast.literal_eval(line)
                    utterances.append(dict_obj)

    # Handle the final block if file didn't end with "--------------------------------------------------"
    if current_block and 'trial' in current_block and 'quadrant' in current_block and 'object_name' in current_block and len(utterances) > 0:
        utterances = sorted(utterances, key=lambda x: x['Utterance'])
        key = (current_block['trial'], current_block['quadrant'], current_block['object_name'])
        blocks_dict[key] = {
            'trial': current_block['trial'],
            'quadrant': current_block['quadrant'],
            'object_name': current_block['object_name'],
            'utterances': utterances
        }

    return blocks_dict


##############################################################################
# 2. Helper to extract "Current Status" in correct order for plotting.
##############################################################################
def get_status_sequence(block_dict):
    """
    Given a block dict of the form:
      {
        'trial': str,
        'quadrant': str,
        'object_name': str,
        'utterances': [list of dicts, sorted by 'Utterance']
      }
    returns two lists:
      - statuses in the order of utterance (e.g., ['UI','A','A','IF',...])
      - the utterance numbers (e.g., [1,2,3,4,...]) for optional x-axis usage
    """
    if not block_dict:
        return [], []
    utterances = sorted(block_dict['utterances'], key=lambda x: x['Utterance'])
    statuses = [u['Current Status'] for u in utterances]
    utt_numbers = [u['Utterance'] for u in utterances]
    return statuses, utt_numbers


##############################################################################
# 3. The main logic: read files, loop over blocks, plot actual vs. predicted
##############################################################################
def main():
    # A) Parse both text files
    truth_file = "true_testing_data.txt"
    pred_file  = "predicted_output_only_changed.txt"

    truth_blocks = parse_input_file(truth_file)
    pred_blocks  = parse_input_file(pred_file)

    # B) Desired status order for numeric mapping (adjust to your actual categories if needed)
    desired_ordered_labels = ['UI', 'F', 'A', 'IF']
    status_to_index = {status: i for i, status in enumerate(desired_ordered_labels)}

    # C) Create an output directory for the plots (optional)
    output_dir = "visualizations_all_objects"
    os.makedirs(output_dir, exist_ok=True)

    # D) Loop over every block in `truth_blocks`
    #    (We only plot if the same (trial, quadrant, object) also appears in `pred_blocks`)
    common_keys = set(truth_blocks.keys()).intersection(set(pred_blocks.keys()))
    if not common_keys:
        print("No common (trial, quadrant, object) found in both files.")
        return

    for key in common_keys:
        # key is (trial, quadrant, object_name)
        trial, quadrant, object_name = key

        truth_block = truth_blocks[key]
        pred_block  = pred_blocks[key]

        # E) Extract the sequences
        y_actual_labels, _ = get_status_sequence(truth_block)
        y_pred_labels, _   = get_status_sequence(pred_block)

        # Use the minimum length if there's a mismatch
        n = min(len(y_actual_labels), len(y_pred_labels))
        y_actual_labels = y_actual_labels[:n]
        y_pred_labels   = y_pred_labels[:n]
        utt_numbers     = list(range(1, n+1))

        # F) Convert statuses to numeric indices
        y_actual_idx = [status_to_index.get(s, -1) for s in y_actual_labels]
        y_pred_idx   = [status_to_index.get(s, -1) for s in y_pred_labels]

        # G) Plot
        plt.figure(figsize=(8, 5))
        plt.plot(utt_numbers, y_actual_idx, marker='o', label='Actual')
        plt.plot(utt_numbers, y_pred_idx, marker='x', linestyle='--', label='Predicted')

        plt.title(f'Cognitive Status Over Time\nTrial: {trial}, Q{quadrant}, Object: {object_name}')
        plt.xlabel('Utterance Index')
        plt.ylabel('Cognitive Status')
        plt.xticks(utt_numbers)
        plt.yticks(range(len(desired_ordered_labels)), desired_ordered_labels)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # H) Save figure (use a unique file name)
        safe_object_name = object_name.replace('/', '_').replace('\\', '_')  # Clean up just in case
        out_filename = f"Trial_{trial}_Quadrant_{quadrant}_Object_{safe_object_name}.png"
        out_path = os.path.join(output_dir, out_filename)
        plt.savefig(out_path)
        plt.close()

        print(f"Saved {out_path}")

    print(f"\nAll plots saved to: {output_dir}")


##############################################################################
# 4. Run the script
##############################################################################
if __name__ == "__main__":
    main()
