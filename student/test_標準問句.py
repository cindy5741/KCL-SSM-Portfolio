import pandas as pd

# 讀取資料
test_df = pd.read_csv("test.csv")  # 包含 text, category
zhongxin_df = pd.read_csv("中心句結果.csv")  # 包含 category_id, category, text

# 建立 category -> 標準問句 對照表
category_to_std = dict(zip(zhongxin_df['category'], zhongxin_df['text']))

# 將 test_df 的 category 替換成標準問句
test_df['category'] = test_df['category'].map(category_to_std)

# 輸出成新的 CSV 或 Excel
test_df.to_csv("test_with_standard_question.csv", index=False)
# 或輸出 Excel
test_df.to_excel("test_with_standard_question.xlsx", index=False)

print("已完成將 category 轉成標準問句")
