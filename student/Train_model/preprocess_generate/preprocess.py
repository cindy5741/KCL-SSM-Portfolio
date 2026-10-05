import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


def read_data(data_dir, has_val=True):
    '''
    data_dir: str, the data directory path
    has_val: boolean, if True, read train/val/test data; else read train/test data
    '''
    if data_dir[-1] != '/':
        data_dir = data_dir + '/'

    train_df = pd.read_csv(data_dir + 'train.tsv', sep='\t').dropna().reset_index(drop=True)
    test_df = pd.read_csv(data_dir + 'test.tsv', sep='\t').dropna().reset_index(drop=True)

    if has_val:
        val_df = pd.read_csv(data_dir + 'val.tsv', sep='\t').dropna().reset_index(drop=True)
        return train_df, val_df, test_df
    else:
        return train_df, test_df


if __name__ == '__main__':

    df = pd.read_excel('../preprocess_data/after_center_usual_questioin.xlsx')
    df = df.replace({r'\t': ' ', r'\n': ' '}, regex=True)
    df = df.dropna().reset_index(drop=True)

    stratified_split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, temp_idx = next(stratified_split.split(df, df['LABEL']))
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(os.path.join(output_dir, 'train.tsv'), sep='\t', index=False)
    val_df.to_csv(os.path.join(output_dir, 'val.tsv'), sep='\t', index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.tsv'), sep='\t', index=False)

    print("✅ Data splitting completed and saved as train.tsv / val.tsv / test.tsv")
