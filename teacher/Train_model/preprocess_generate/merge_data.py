import pandas as pd


df = pd.read_excel('../preprocess_data/AI擬真問句_Q1-Q250_250630.xlsx')
df_cleaned = df[df['擬真問句'] != '擬真問句']

df_cleaned = df_cleaned[
    df_cleaned['擬真問句'].notna() & df_cleaned['標準問句'].notna() & 
    (df_cleaned['擬真問句'].str.strip() != '') & (df_cleaned['標準問句'].str.strip() != '')
]

# Group by the "standard question" and assign a unique group ID starting from 1
# Create a dictionary that maps each standard question to its corresponding group ID

standard_question_to_id = {
    question: idx + 1 for idx, question in enumerate(df_cleaned['標準問句'].unique())
}

df_cleaned['編號'] = df_cleaned['標準問句'].map(standard_question_to_id)

# Rearrange the column order
new_df = df_cleaned[['編號', '擬真問句', '標準問句']].copy()
new_df = new_df.sort_values(by=['編號']).reset_index(drop=True)


print(new_df.head())
new_df.to_excel('../preprocess_data/final_merge_data_251_label.xlsx', index=False)
