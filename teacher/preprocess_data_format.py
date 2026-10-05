# import pandas as pd
# import re

# # === 1️⃣ 讀取中心句結果和原始 Excel ===
# df_center = pd.read_csv("中心句結果.csv", dtype={"category_id": str})
# df_raw = pd.read_excel("LLM生成後訓練資料.xlsx", sheet_name=0)

# data = []
# debug_info = []
# trimmed_rows = []
# seen_questions = set()

# for idx, row in df_raw.iterrows():

#     problem_text = str(row.get("question", ""))

#     # ---- 修正後 Regex：抓下一行資料 ----
#     id_match = re.search(r"#category_id\s*\n\s*(\d+)", problem_text)
#     cat_match = re.search(r"#category\s*\n\s*([^\n#]+)", problem_text)
#     text_match = re.search(r"#text\s*\n\s*(.+?)(?=\n#text|\Z)", problem_text, re.S)

#     if not id_match or not cat_match or not text_match:
#         debug_info.append({
#             "列號": idx + 2,
#             "category_id": "",
#             "問題": "找不到 category_id 或 category 或 #text"
#         })
#         continue

#     category_id = id_match.group(1).strip()
#     category = cat_match.group(1).strip()
#     original_text = text_match.group(1).strip()

#     # ---- 找標準問句 ----
#     center_row = df_center[df_center["category_id"] == category_id]
#     if center_row.empty:
#         debug_info.append({
#             "列號": idx + 2,
#             "category_id": category_id,
#             "問題": "找不到對應中心句"
#         })
#         continue

#     standard_question = center_row.iloc[0]["text"]

#     # ---- 拆 AI回答 ----
#     answers_text = str(row.get("擬真問句", ""))
#     answers = [a.strip() for a in answers_text.split("\n") if a.strip()]

#     # === ✨ 超過 10 句 → 修剪 ===
#     if len(answers) > 10:
#         trimmed_rows.append({
#             "列號": idx + 2,
#             "原本句數": len(answers),
#             "修剪後句數": 10,
#             "前200字": answers_text[:200]
#         })
#         answers = answers[:10]

#     # === ✨ 少於 10 句 → debug ===
#     if len(answers) < 10:
#         debug_info.append({
#             "列號": idx + 2,
#             "category_id": category_id,
#             "實際生成句數": len(answers),
#             "AI回答原文": answers_text[:200]
#         })

#     # ---- 寫入最終輸出 ----
#     for ans in answers:

#         # 🎯 重複句 → same!
#         if ans in seen_questions:
#             ans_to_save = "same!"
#         else:
#             ans_to_save = ans
#             seen_questions.add(ans)

#         data.append({
#             "編號": category_id,
#             "標準問句": standard_question,
#             "擬真問句": ans_to_save,
#             "text": original_text
#         })

# # === 2️⃣ 輸出主要結果 ===
# df_output = pd.DataFrame(data)
# df_output.to_excel("檢查20萬句llm生成擬真問句_去重.xlsx", index=False)

# # === 3️⃣ 偵錯報告（不足 10 行）===
# if debug_info:
#     df_debug = pd.DataFrame(debug_info)
#     df_debug.to_excel("偵錯報告.xlsx", index=False)

# # === 4️⃣ 修剪報告 ===
# if trimmed_rows:
#     df_trim = pd.DataFrame(trimmed_rows)
#     df_trim.to_excel("被修剪的列.xlsx", index=False)

# # === 5️⃣ 統計 ===
# expected_total = len(df_raw) * 10
# actual_total = len(df_output)
# missing = expected_total - actual_total

# print("✅ 匯出完成！")
# print(f"📊 理論筆數：{expected_total}")
# print(f"📈 實際筆數：{actual_total}")
# print(f"⚠️ 缺少筆數：{missing}")

# if debug_info:
#     print(f"⚠️ 有 {len(debug_info)} 筆資料生成不足 10 句，詳見 偵錯報告.xlsx")

# if trimmed_rows:
#     print(f"✂️ 有 {len(trimmed_rows)} 筆資料超過 10 句已被自動修剪，詳見 被修剪的列.xlsx")
import pandas as pd

# 1️⃣ 讀取兩個檔案
df1 = pd.read_excel("檢查20萬句llm生成擬真問句_去重.xlsx")
df2 = pd.read_excel("要重新生的.xlsx")

# 2️⃣ 刪掉 df1 中「擬真問句」是空白的資料
df1 = df1[df1['擬真問句'].notna() & (df1['擬真問句'].astype(str).str.strip() != '')]

# 3️⃣ 合併 df1 和 df2
df_combined = pd.concat([df1, df2], ignore_index=True)

# 4️⃣ 按照編號（假設欄位名為 '編號'）和 text 排序
df_combined = df_combined.sort_values(by=['編號', 'text']).reset_index(drop=True)

# 5️⃣ 找到擬真問句重複的，把重複值設為空白（保留第一筆）
duplicated_mask = df_combined.duplicated(subset=['擬真問句'], keep='first')
df_combined.loc[duplicated_mask, '擬真問句'] = ''

# 6️⃣ 儲存結果
df_combined.to_excel("合併去重結果.xlsx", index=False)

print("✅ 合併完成，重複擬真問句已設為空白，結果已儲存到 '合併去重結果.xlsx'")
