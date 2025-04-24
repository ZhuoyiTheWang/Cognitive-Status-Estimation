import pandas as pd
import os

def update_obj_with_cur_status(trial):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(f'data/Cog Status/{trial}.xlsx - Cognitive Status.csv')

    # Define the logic for the new column 'current cog status'
    def determine_cog_status(row):
        if row['in'] == 'A,IF':
            return 'IF'
        elif row['in'] == 'IF':
            return 'IF'
        elif row['in'] == 'A':
            return 'A'
        elif row['out'] == 'IF':
            return 'A'
        elif row['out'] == 'A':
            return 'F'
        return None  # Default value if no conditions match

    # Apply the logic to each row
    df['Current Status'] = df.apply(determine_cog_status, axis=1)
    df = df.drop(['in', 'out'], axis=1)

    return df

for i in range (1, 12):
    if os.path.exists(f'data/Cog Status/P{i}.xlsx - Cognitive Status.csv'):
        updated_df = update_obj_with_cur_status(f'P{i}')
        updated_df.to_csv(f'data/Cog Status (Cur Status)/P{i} - Cognitive Status.csv', index=False)