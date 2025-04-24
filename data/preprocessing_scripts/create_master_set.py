import pandas as pd
from pathlib import Path

# Specify the directory path
directory = Path('data/Coding Isolated')

# List all files in the directory
all_files = [f for f in directory.rglob('*') if f.is_file()]
trials = []

# Print or process the list of files
for file in all_files:
    file = str(file)
    file = file.split('\\')[2].split(' ')[0]
    trials.append(file)


def create_master_set(trial):
    print(trial)
    # Read the CSV file into a dataframe
    objects = pd.read_csv(f'data/Objects/{trial}.xlsx - Objects.csv', dtype=str)

    cog_status = pd.read_csv(f'data/Cog Status (Cur Status)/{trial} - Cognitive Status.csv', dtype=str)
    cog_status = cog_status.dropna(subset=['Quadrant', 'Utterance'])
    cog_status_int_columns = ['Quadrant', 'Utterance']
    for col in cog_status_int_columns:
        cog_status[col] = pd.to_numeric(cog_status[col], errors='coerce')

    coding = pd.read_csv(f'data/Coding Isolated/{trial} - Coding.csv', dtype=str)
    coding = coding.dropna(subset=['Quadrant', 'Utterance Number'])
    coding_int_columns = ['Quadrant', 'Utterance Number']
    for col in coding_int_columns:
        coding[col] = pd.to_numeric(coding[col], errors='coerce')

    # Initialize an empty list to store the new rows
    new_rows = []
    object_bounding_dict = {'Q2': {}, 'Q3': {}}

    # Loop through quadrants 2 and 3
    for quadrant in range(2, 4):
        # Loop through utterances 1 to 15
        for utterance in range(1, 16):
            # Iterate through the dataframe
            for index, row in objects.iterrows():
                # Append the new row to the list
                new_rows.append({
                    "Quadrant": quadrant,
                    "Utterance": utterance,
                    "Object": row["Objects"],
                    "Current Status": 'F' if row["Objects"].split('-')[0] == ('Q' + str(quadrant)) else "UI",
                    "Mentioned": "N",
                    "Gesture": "N",
                    "Grammatical Role": "N",
                    "Previous Status": 'F' if row["Objects"].split('-')[0] == ('Q' + str(quadrant)) else "UI"
                })

    # Create a new dataframe with the expanded rows
    expanded_df = pd.DataFrame(new_rows)

    for index, row in cog_status.iterrows():
        if pd.notna(row['Current Status']):
            if pd.notna(row['Bound to']):
                condition = expanded_df['Object'] == row['Bound to'].split(' ')[0]
                object_bounding_dict['Q' + str(int(float(row['Quadrant'])))][row['object']] = row['Bound to'].split(' ')[0]
            else:
                condition = expanded_df['Object'] == row['object']

            quadrant = int(float(row['Quadrant']))
            utterance = int(float(row['Utterance']))
            # Iterate through the filtered rows and modify the original dataframe
            for expanded_index, expanded_row in expanded_df[condition].iterrows():
                # Modify the original dataframe at the current index
                if expanded_df.at[expanded_index, 'Quadrant'] == quadrant and expanded_df.at[expanded_index, 'Utterance'] >= utterance:
                    expanded_df.at[expanded_index, 'Current Status'] = row['Current Status']
                if expanded_df.at[expanded_index, 'Quadrant'] == quadrant and expanded_df.at[expanded_index, 'Utterance'] > utterance:
                    expanded_df.at[expanded_index, 'Previous Status'] = row['Current Status']

    for index, row in coding.iterrows():
        if str(row['Object referenced']).split('-')[0] == 'New':
            if str(row['Object referenced']) in object_bounding_dict['Q' + str(int(float(row['Quadrant'])))]:
                row['Object referenced'] = object_bounding_dict['Q' + str(int(float(row['Quadrant'])))][row['Object referenced']]
        
        # Use boolean indexing to find the original index
        condition = (
            (expanded_df['Quadrant'] == row['Quadrant']) &
            (expanded_df['Utterance'] == row['Utterance Number']) &
            (expanded_df['Object'] == row['Object referenced'])
        )

        # Get the original index(es)
        matching_index = expanded_df[condition].index

        if matching_index.empty:
            print(f"{row['Quadrant'], row['Utterance Number'], row['Object referenced']}")
        else:
            expanded_df.at[matching_index[0], 'Gesture'] = row['Gesture Used']
            expanded_df.at[matching_index[0], 'Mentioned'] = 'M'
            
            if not pd.isnull(row['grammatical role']):
                expanded_df.at[matching_index[0], 'Grammatical Role'] = row['grammatical role']

    print(object_bounding_dict)

    # Save the new dataframe to a new CSV file
    expanded_df.to_csv(f'data/Master Dataset/{trial}.csv', index=False)

for trial in trials:
    create_master_set(trial)