import pandas as pd

# 讀取 CSV
train_df = pd.read_csv("train_標準問句.csv")      # columns: text, category
zhongxin_df = pd.read_csv("中心句結果.csv")       # columns: category_id, category, text

# 建立 category (train.csv.category / 標準問句) -> category_id 的對應
category_to_id = dict(zip(zhongxin_df['text'], zhongxin_df['category_id']))
# 建立 category (train.csv.category / 標準問句) -> 中心句結果 text 的對應
category_to_standard_text = dict(zip(zhongxin_df['text'], zhongxin_df['text']))

# 用 train.csv 的 category 去對應中心句結果，拿到編號和標準問句
train_df['編號'] = train_df['category'].map(category_to_id)
train_df['標準問句'] = train_df['category'].map(category_to_standard_text)
train_df.rename(columns={'text':'擬真問句'}, inplace=True)

# 選擇最終欄位順序
final_df = train_df[['編號', '擬真問句', '標準問句']]

# 輸出 Excel
final_df.to_excel("真實訓練資料.xlsx", index=False)
print("已完成生成：真實訓練資料.xlsx")
