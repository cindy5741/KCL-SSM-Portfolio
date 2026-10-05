import pandas as pd
import re

# === 1️⃣ 讀取中心句結果和原始 Excel ===
df_center = pd.read_csv("中心句結果.csv", dtype={"category_id": str})
df_raw = pd.read_excel("train生成結果.xlsx", sheet_name="批次聊天結果")

data = []           # 儲存最終輸出
debug_info = []     # 儲存偵錯資訊
seen_questions = set()  # 記錄已出現的擬真問句，用於去重

for idx, row in df_raw.iterrows():
    problem_text = str(row.get("問題", ""))
    
    # ---- 取出 category_id, category, #text ----
    id_match = re.search(r"#category_id\s*(\d+)", problem_text)
    cat_match = re.search(r"#category\s*([^\n#]+)", problem_text)
    text_match = re.search(r"#text\s*(.+)", problem_text)

    if not id_match or not cat_match or not text_match:
        debug_info.append({
            "列號": idx + 2,
            "category_id": "",
            "問題": "找不到 category_id 或 category 或 #text"
        })
        continue

    category_id = str(id_match.group(1).strip())
    category = cat_match.group(1).strip()
    original_text = text_match.group(1).strip()  # <- 原始 #text

    # ---- 找標準問句 ----
    center_row = df_center[df_center["category_id"] == category_id]
    if center_row.empty:
        debug_info.append({
            "列號": idx + 2,
            "category_id": category_id,
            "問題": "找不到對應中心句"
        })
        continue

    standard_question = center_row.iloc[0]["text"]

    # ---- 拆解 AI回答欄，每行當一個擬真問句 ----
    answers_text = str(row.get("AI回答", ""))
    answers = [a.strip() for a in answers_text.split("\n") if a.strip()]
    count = len(answers)

    if count < 10:
        debug_info.append({
            "列號": idx + 2,
            "category_id": category_id,
            "實際生成句數": count,
            "AI回答原文": answers_text[:200]  # 截取前200字方便檢查
        })

    for ans in answers:
        # 如果擬真問句已出現過，就清空該格
        if ans in seen_questions:
            ans_to_save = ""
        else:
            ans_to_save = ans
            seen_questions.add(ans)

        data.append({
            "編號": category_id,
            "標準問句": standard_question,
            "擬真問句": ans_to_save,
            "text": original_text  # <- 記錄原始 #text
        })

# === 2️⃣ 輸出主要結果 ===
df_output = pd.DataFrame(data)
df_output.to_excel("10萬句llm生成擬真問句_去重.xlsx", index=False)

# === 3️⃣ 輸出偵錯報告 ===
if debug_info:
    df_debug = pd.DataFrame(debug_info)
    df_debug.to_excel("偵錯報告.xlsx", index=False)

# === 4️⃣ 統計資訊 ===
expected_total = len(df_raw) * 10
actual_total = len(df_output)
missing = expected_total - actual_total

print("✅ 匯出完成！")
print(f"📊 理論筆數：{expected_total}")
print(f"📈 實際筆數：{actual_total}")
print(f"⚠️ 缺少筆數：{missing}")
if debug_info:
    print(f"⚠️ 有 {len(debug_info)} 筆資料生成不足 10 句或抓不到問句，詳見 偵錯報告.xlsx")
