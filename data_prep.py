import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from logger import log_message
from grouping import replace_values
from config import RANDOM_STATE

def prepare_balance_and_split_data(df, model, block_group):
    df_group = df[df['BlockGroup'] == block_group].copy()
    if df_group.empty:
        raise ValueError(f"No data available for group '{block_group}'.")

    if model == 2:
        log_message("Excluding rows where 'MTSChosen' equals DCT2_DCT2 or SKIP")
        df_group = df_group[~df_group['MTSChosen'].isin([0, 1])]

    df_group['MTSChosen'] = df_group['MTSChosen'].apply(lambda v: replace_values(v, model))

    if df_group['MTSChosen'].nunique() < 2:
        raise ValueError(f"Only one class present for group '{block_group}'.")

    df_group['balance_key'] = list(zip(
        df_group['MTSChosen'],
        df_group['cuQP'],
        df_group['FrameWidth'],
        df_group['FrameHeight']
    ))

    min_samples_per_key = df_group['balance_key'].value_counts().min()
    if min_samples_per_key < 1:
        raise ValueError(f"Group '{block_group}' does not have sufficient combinations.")

    balanced = pd.concat([
        resample(g, replace=False, n_samples=min_samples_per_key, random_state=RANDOM_STATE)
        for _, g in df_group.groupby('balance_key')
    ])

    log_message(f"Group '{block_group}' balanced. Total: {len(balanced)} samples.")

    X = balanced.drop(columns=['MTSChosen', 'BlockGroup', 'VideoName', 'balance_key'], errors='ignore')
    y = balanced['MTSChosen']

    return train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
