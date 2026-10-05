# import pandas as pd

# # 讀取 Excel
# df = pd.read_excel("合併去重結果.xlsx")

# # 查看總筆數
# print(f"總筆數: {len(df)}")

# # 刪除完全重複的行，保留第一筆
# df_no_duplicates = df.drop_duplicates(keep='first')

# # 查看刪除後的筆數
# print(f"刪除重複後總筆數: {len(df_no_duplicates)}")

# # 將去重後的資料輸出到新的 Excel
# df_no_duplicates.to_excel("合併去重結果.xlsx", index=False)

# print("✅ 已生成去重後的 Excel：合併去重結果.xlsx")

import pandas as pd

# 讀取 Excel
df = pd.read_excel("合併去重結果_AI已填補.xlsx")

# 查看總筆數
print(f"📊 總筆數: {len(df)}")

# === 檢查重複值 ===
# 假設你要檢查的是整列重複（所有欄位都一樣）
duplicate_rows = df[df.duplicated(keep=False)]
num_duplicates = len(duplicate_rows)
print(f"🔁 重複句數: {num_duplicates}")

# === 檢查空白值 ===
# 找出任一欄為空的列（若只想檢查特定欄可改 df['欄名'].isna()）
blank_rows = df[df["text"].isna() | (df["text"].astype(str).str.strip() == "")]
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

# import pandas as pd

# # 讀檔
# df = pd.read_excel("合併去重結果.xlsx")  # 或 pd.read_excel

# # 計算每個標準問句對應的 text 筆數
# counts = df.groupby('text')['標準問句'].count().reset_index()
# counts.rename(columns={'標準問句':'count'}, inplace=True)

# # 找出不是10次的 text
# wrong = counts[counts['count'] != 10]

# too_few = wrong[wrong['count'] < 10]
# too_many = wrong[wrong['count'] > 10]

# print(f"總 text 數量: {len(counts)}")
# print(f"text 不是10次的數量: {len(wrong)}")
# print(f"少於10次: {len(too_few)}")
# print(f"多於10次: {len(too_many)}")
# print(wrong)


# import pandas as pd

# # 1. 讀檔
# input_file = "合併去重結果.xlsx"
# output_file = "補齊10筆結果.xlsx"

# try:
#     df = pd.read_excel(input_file)
#     print(f"成功讀取檔案，原始筆數: {len(df)}")
# except FileNotFoundError:
#     print(f"找不到檔案: {input_file}")
#     exit()

# # 2. 計算每個 text 出現次數
# counts = df.groupby('text').size().reset_index(name='count')

# # 3. 找出少於 10 次的 text
# target_count = 10
# too_few = counts[counts['count'] < target_count]

# if too_few.empty:
#     print("所有類別皆已達到 10 筆，無需補齊。")
# else:
#     print(f"發現 {len(too_few)} 個類別少於 {target_count} 筆，開始補齊...")

#     # 4. 補足缺少的筆數
#     new_rows = []
    
#     # 確保編號欄位存在且為數值，若無則設為 0
#     if '編號' in df.columns:
#         current_max_id = df['編號'].max()
#     else:
#         current_max_id = 0

#     for _, row in too_few.iterrows():
#         text = row['text']
#         cnt = row['count']
#         needed = target_count - cnt
        
#         # 取出該 text 對應的第一個標準問句 (假設同一 text 對應的標準問句都一樣)
#         std_question = df[df['text'] == text]['標準問句'].iloc[0]

#         for _ in range(needed):
#             current_max_id += 1
#             new_rows.append({
#                 '編號': current_max_id,
#                 '標準問句': std_question,
#                 '擬真問句': '',  # 留空待填
#                 'text': text
#             })

#     # 5. 加回原本的 df 並存檔
#     if new_rows:
#         new_df = pd.DataFrame(new_rows)
#         # 使用 concat 合併，確保欄位對齊
#         df = pd.concat([df, new_df], ignore_index=True)
        
#         df.to_excel(output_file, index=False)
#         print(f"--- 處理完成 ---")
#         print(f"總共新增了 {len(new_rows)} 筆空白資料")
#         print(f"結果已儲存至: {output_file}")
#     else:
#         print("沒有產生新的資料行。")