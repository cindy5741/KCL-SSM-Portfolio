# import pandas as pd

# # 讀取 Excel
# df = pd.read_excel("給gemini跑_AI填補完成10萬句llm生成擬真問句.xlsx")

# # 查看總筆數
# print(f"總筆數: {len(df)}")

# # 刪除完全重複的行，保留第一筆
# df_no_duplicates = df.drop_duplicates(keep='first')

# # 查看刪除後的筆數
# print(f"刪除重複後總筆數: {len(df_no_duplicates)}")

# # 將去重後的資料輸出到新的 Excel
# df_no_duplicates.to_excel("10萬句llm生成擬真問句_去重.xlsx", index=False)

# print("✅ 已生成去重後的 Excel：10萬句llm生成擬真問句_去重.xlsx")

import pandas as pd

# 讀取 Excel
df = pd.read_excel("10萬句llm生成擬真問句_去重.xlsx")

# 查看總筆數
print(f"📊 總筆數: {len(df)}")

# === 檢查重複值 ===
# 假設你要檢查的是整列重複（所有欄位都一樣）
duplicate_rows = df[df.duplicated(keep=False)]
num_duplicates = len(duplicate_rows)
print(f"🔁 重複句數: {num_duplicates}")

# === 檢查空白值 ===
# 找出任一欄為空的列（若只想檢查特定欄可改 df['欄名'].isna()）
blank_rows = df[df.isna().any(axis=1)]
num_blank = len(blank_rows)
print(f"⚠️ 空白句數: {num_blank}")

# === 若想看重複或空白的內容，可輸出方便檢查 ===
# if num_duplicates > 0:
#     duplicate_rows.to_excel("重複句檢查.xlsx", index=False)
#     print("📁 已輸出重複句檢查.xlsx")

# if num_blank > 0:
#     blank_rows.to_excel("空白句檢查.xlsx", index=False)
#     print("📁 已輸出空白句檢查.xlsx")

print("✅ 檢查完成")
