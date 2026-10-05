import pandas as pd

# === 1. 讀取兩份資料 ===
a = pd.read_excel("FAQ_13萬筆資料_含1306筆真實.xlsx")  # 有「編號」「擬真問句」「標準問句」
b = pd.read_excel("251006_編號251真實問句生成.xlsx")  # 有「知識編號」「生成問句」

# === 2. 統一欄位名稱 ===
b = b.rename(columns={"知識編號": "編號", "生成問句": "擬真問句"})

# === 3. 用 a.xlsx 的標準問句對應上 b.xlsx ===
b = pd.merge(b, a[["編號", "標準問句"]], on="編號", how="left")

# === 4. 合併 a 與 b 的資料 ===
combined = pd.concat([a, b], ignore_index=True)
# combined = combined.dropna()
# === 5. 只保留需要的欄位 ===
combined = combined[["編號", "擬真問句", "標準問句"]]
combined["編號"] = combined["編號"].astype(str).str.replace(".0", "", regex=False)
combined = combined.drop_duplicates(subset=["編號", "擬真問句", "標準問句"], keep="first")

# === 6. 輸出 ===
combined.to_excel("FAQ_13萬筆+真實問句訓練素材1306筆_加上新類別(真實與生成).xlsx", index=False)
combined.to_csv("FAQ_13萬筆+真實問句訓練素材1306筆_加上新類別(真實與生成).csv", index=False, encoding='utf-8')
print("✅ 合併完成！已輸出 xlsx")

print(combined[combined.isna().any(axis=1)])
