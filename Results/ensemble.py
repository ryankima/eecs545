import pandas as pd

models = ["coatnet", "efficientnet", "deit"]
for model in models:
    # Step 1: Read the two CSV files into DataFrames
    df1 = pd.read_csv(f'baseline/scores_{model}_ISIC_2024_Training_Input.csv')
    df2 = pd.read_csv(f'ST/scores_{model}_ISIC_2024_Training_Input_mixed_styled.csv')

    merged_df = pd.merge(df1, df2, on='isic_id', suffixes=('_base', '_st'))
    merged_df['prediction'] = (merged_df['prediction_base'] + merged_df['prediction_st']) / 2
    merged_df['target'] = merged_df['target_base']
    merged_df = merged_df.drop(columns=['prediction_base', 'prediction_st', 'target_base', 'target_st'])
    merged_df.to_csv(f'ensemble/scores_{model}.csv', index=False)