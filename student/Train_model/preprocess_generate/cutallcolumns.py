import pandas as pd

df = pd.read_excel('../preprocess_data/final_merge_data_251_label.xlsx')

new_rows = []

for index, row in df.iterrows():
    knowledge = row['編號']
    standard = row['標準問句']
    questions = str(row['擬真問句']).split('\n')  
    for q in questions:
        q = q.strip()

        new_rows.append({'QUESTION': standard, 'SAMPLE': q, 'LABEL' : knowledge})
new_df = pd.DataFrame(new_rows)


new_df.to_excel('../preprocess_data/after_center_usual_questioin.xlsx', index=False)

