import pandas as pd

# 讀取原始 Excel
df = pd.read_excel("[新增轉接與結束通話指令]裁決中心常見問題_20250926.xlsx")

# 篩選出標準問句 = "結束通話"
df_target = df[df["標準問句"] == "結束通話"].copy()

# 編號設定
standard_id = 251
standard_question = "結束通話"

rows = []

# 遍歷篩選後的每一列，處理"擬真問句"
for text in df_target["問題"].dropna().astype(str):
    # 用換行拆分
    for s in text.split("\n"):
        s = s.strip()
        if s:  # 過濾空字串
            rows.append({
                "編號": standard_id,
                "擬真問句": s,
                "標準問句": standard_question
            })

# 整理成 DataFrame
result = pd.DataFrame(rows)

# 輸出新的 Excel
result.to_excel("結束通話_整理後.xlsx", index=False)
result.to_csv("結束通話_整理後.csv", index=False, encoding="utf-8-sig")

print("完成！已輸出結束通話_整理後.xlsx")
