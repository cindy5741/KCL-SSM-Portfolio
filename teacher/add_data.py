import pandas as pd

# 讀取兩個 Excel
df1 = pd.read_excel("FAQ_13萬筆資料.xlsx")
df2 = pd.read_excel("FAQ_1306筆真實資料.xlsx")
df3 = pd.read_excel("結束通話_整理後.xlsx")


# 合併
df = pd.concat([df1, df2, df3], ignore_index=True)
df=df.dropna()
# 找出重複的列（全部欄位都相同才算重複）
duplicates = df[df.duplicated(keep=False)]

print(duplicates)

df = df.drop_duplicates()


# 存成新的 Excel
df.to_excel("FAQ_13萬筆資料_含1306筆真實.xlsx", index=False)
df.to_csv("FAQ_13萬筆資料_含1306筆真實.csv", index=False, encoding="utf-8-sig")

# 把重複的列另存成一個檔案
duplicates.to_excel("FAQ_重複列.xlsx", index=False)
duplicates.to_csv("FAQ_重複列.csv", index=False, encoding="utf-8-sig")