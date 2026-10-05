import pandas as pd

df = pd.read_excel('../preprocess_data/after_center_usual_questioin.xlsx')

df = df.drop(columns=['QUESTION'])

df.to_csv('../../data/all_data_label.tsv', sep='\t', index=False)