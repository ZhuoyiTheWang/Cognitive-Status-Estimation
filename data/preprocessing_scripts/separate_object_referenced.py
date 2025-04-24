import pandas as pd
import os

def separate_references(trial):
    # Read your CSV file
    df = pd.read_csv(f'data/Coding/{trial}.xlsx - Coding.csv')

    # Split the "Object referenced" column where there are multiple objects
    df_expanded = df.assign(**{'Object referenced': df['Object referenced'].str.split(', ')})

    # Explode the DataFrame to expand the lists into individual rows
    df_exploded = df_expanded.explode('Object referenced')

    # Sort values to prioritize gestures that are not 'N'
    df_exploded['Gesture Used'] = df_exploded['Gesture Used'].fillna('N')  # Handle any NaN values in Gesture Used

    # Drop duplicates keeping the first (i.e., the non-'N' gestures if available)
    df_final = df_exploded.drop_duplicates(subset=['Quadrant', 'Utterance Number', 'Object referenced', 'grammatical role'], keep='first')

    return df_final


for i in range (1, 12):
    if os.path.exists(f'data/Coding/P{i}.xlsx - Coding.csv'):
        reference_separated_df = separate_references(f'P{i}')
        reference_separated_df.to_csv(f'data/Coding Isolated/P{i} - Coding.csv', index=False)