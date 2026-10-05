import pandas as pd

# ===== 1️⃣ 定義文字清理函數 =====
def clean_text(s):
    if pd.isna(s):
        return s
    return (
        s.strip()                    # 去掉前後空白
         .replace("\r", "")           # 去掉換行
         .replace("\n", "")
         .replace("\t", "")           # 去掉制表符
         .replace("\u3000", " ")      # 全形空格
         .replace("\xa0", " ")        # 不可見空白
    )

# ===== 2️⃣ 讀取 1031句 作為知識編號對照表 =====
df1 = pd.read_csv("1031句.csv")
df1["標準問句"] = df1["標準問句"].apply(clean_text)
df1 = df1.rename(columns={"問題": "擬真問句"})
df1 = df1[["知識編號", "擬真問句", "標準問句"]]

# 建立標準問句 -> 知識編號 mapping
mapping = dict(zip(df1["標準問句"], df1["知識編號"]))

# ===== 3️⃣ 處理 56句、117句、102句 =====
def process_csv(file_path, std_col, real_col):
    df = pd.read_csv(file_path)
    df = df.rename(columns={std_col: "標準問句", real_col: "擬真問句"})
    df["標準問句"] = df["標準問句"].apply(clean_text)
    df["擬真問句"] = df["擬真問句"].apply(clean_text)
    df["知識編號"] = df["標準問句"].map(mapping)
    df = df[["知識編號", "擬真問句", "標準問句"]]
    return df

df2 = process_csv("56句.csv", "標註問題", "真實問句(來自電話逐字稿)")
df3 = process_csv("117句.csv", "標註問題", "真實問句(來自電話逐字稿)")
df4 = process_csv("102句.csv", "標註問題", "真實問句(來自電話逐字稿)")

# ===== 4️⃣ 合併四個 CSV =====
df_all = pd.concat([df1, df2, df3, df4], ignore_index=True)
df_all["知識編號"] = df_all["知識編號"].astype("Int64")  # 支援缺失值

# 將欄位名稱「知識編號」改成「編號」
df_all = df_all.rename(columns={"知識編號": "編號"})

# ===== 5️⃣ 存檔 =====
df_all.to_excel("FAQ_1306筆真實未生成資料.xlsx", index=False)
df_all.to_csv("FAQ_1306筆真實未生成資料.csv", index=False, encoding="utf-8-sig")

print("完成！合併後共", len(df_all), "筆資料")

# # ===== 6️⃣ 列出無法對應知識編號的行 =====
unmatched = df_all[df_all["編號"].isna()]
if not unmatched.empty:
    print("\n以下為無法對應知識編號的資料：")
    for i, row in unmatched.iterrows():
        print(f"Index {i}: 擬真問句 = {row['擬真問句']}, 標準問句 = {row['標準問句']}")
else:
    print("全部資料都有對應到知識編號！")

# ===== 6️⃣ 統計每個標準問句對應的擬真問句數量 =====
# df_count = df_all.groupby("標準問句")["擬真問句"].count().reset_index()
# df_count = df_count.rename(columns={"擬真問句": "擬真問句數量"})

# # 存成新的 CSV
# df_count.to_csv("標準問句擬真問句統計.csv", index=False, encoding="utf-8-sig")

# print("已生成 CSV：標準問句擬真問句統計.csv，共", len(df_count), "筆標準問句")