import pandas as pd

df = pd.read_excel('../preprocess_data/after_center_usual_questioin.xlsx')

df = df.drop(columns=['SAMPLE'])
df = df.drop_duplicates()

df.to_csv('../../data/question_to_label.tsv', sep='\t', index=False)