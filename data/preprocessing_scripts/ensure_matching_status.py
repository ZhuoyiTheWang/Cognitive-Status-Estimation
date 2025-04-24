import pandas as pd
import os

# Define the conditions for entry and exit
ENTRY_CONDITIONS = ['A', 'IF']
EXIT_CONDITIONS = ['A', 'IF']

def ensure_matching(trial):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(f'data/Cog Status Bounded/{trial} - Cognitive Status.csv')  # Replace 'your_file.csv' with your actual file name

    # Function to check for mismatches between in and out conditions
    def check_in_and_out_mismatches(df):
        # Initialize sets for objects with mismatches
        went_in_but_never_out_A = set()
        went_in_but_never_out_IF = set()
        went_out_without_in_A = set()
        went_out_without_in_IF = set()

        # Group by object to check in/out conditions for each one
        grouped = df.groupby('object')

        for obj, group in grouped:
            in_conditions_met = set()   # Store conditions for "in"
            out_conditions_met = set()  # Store conditions for "out"

            # Loop through each row in the group for the current object
            for _, row in group.iterrows():
                # Get the "in" and "out" values for the current row, and clean them
                in_values = set(str(row['in']).split(','))
                out_values = set(str(row['out']).split(','))

                # Add the "in" and "out" values to the respective sets
                in_conditions_met.update(in_values)
                out_conditions_met.update(out_values)

            # Check for mismatches for A
            if 'A' in in_conditions_met and 'A' not in out_conditions_met:
                went_in_but_never_out_A.add(obj)
            if 'A' not in in_conditions_met and 'A' in out_conditions_met:
                went_out_without_in_A.add(obj)

            # Check for mismatches for IF
            if 'IF' in in_conditions_met and 'IF' not in out_conditions_met:
                went_in_but_never_out_IF.add(obj)
            if 'IF' not in in_conditions_met and 'IF' in out_conditions_met:
                went_out_without_in_IF.add(obj)

        return (went_in_but_never_out_A, went_in_but_never_out_IF,
                went_out_without_in_A, went_out_without_in_IF)

    # Call the function to check mismatches
    (went_in_but_never_out_A, went_in_but_never_out_IF,
    went_out_without_in_A, went_out_without_in_IF) = check_in_and_out_mismatches(df)

    # Output the results for mismatches
    print(f"\n{trial}:")
    if went_in_but_never_out_A:
        print("\tObjects that went in for A but never exited A:", went_in_but_never_out_A)
    if went_in_but_never_out_IF:
        print("\tObjects that went in for IF but never exited IF:", went_in_but_never_out_IF)
    if went_out_without_in_A:
        print("\tObjects that exited A but never went in for A:", went_out_without_in_A)
    if went_out_without_in_IF:
        print("\tObjects that exited IF but never went in for IF:", went_out_without_in_IF)

for i in range (1, 12):
    if os.path.exists(f'data/Cog Status Bounded/P{i} - Cognitive Status.csv'):
        ensure_matching(f'P{i}')