import pandas as pd
import os

def update_obj_with_bound(trial):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(f'data/Cog Status/{trial}.xlsx - Cognitive Status.csv')

    # Function to overwrite 'object' with the first element of 'Bound to' if 'Bound to' is not empty
    def overwrite_object(row):
        if pd.notna(row['Bound to']) and row['Bound to'].strip():  # Check if 'Bound to' is not empty
            return row['Bound to'].split(' ')[0]  # Return the first element of 'Bound to'
        return row['object']  # Otherwise, keep the original 'object'

    # Apply the function to the DataFrame
    df['object'] = df.apply(overwrite_object, axis=1)

    return df

for i in range (1, 12):
    if os.path.exists(f'data/Cog Status/P{i}.xlsx - Cognitive Status.csv'):
        updated_df = update_obj_with_bound(f'P{i}')
        updated_df.to_csv(f'data/Cog Status Bounded/P{i} - Cognitive Status.csv', index=False)