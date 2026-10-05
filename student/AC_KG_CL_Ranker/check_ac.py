import pandas as pd
import ahocorasick
from pathlib import Path
from collections import defaultdict
from build_graph import KnowledgeGraphBuilder
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt_tab")
nltk.download("punkt")
nltk.download("wordnet")

# === 路徑設定 ===
CURRENT_DIR = Path(__file__).resolve().parent

# === 讀取資料 ===
query_df = pd.read_excel(CURRENT_DIR / "test_with_standard_question.xlsx")
label_df = pd.read_csv(CURRENT_DIR / "中心句結果(編號對應標準問句).csv")
label_df = label_df.dropna(subset=["id", "標準問句"])
label_df["id"] = label_df["id"].astype(int)
label_map = dict(zip(label_df["id"], label_df["標準問句"]))

# === 初始化 AC ===
builder = KnowledgeGraphBuilder()
builder.load_data_from_excel(CURRENT_DIR / "中心句關鍵字.xlsx")
builder.build_graph()

AC = ahocorasick.Automaton()
for kw in builder.keyword_to_questions.keys():
    # 將 keyword 也轉成小寫
    AC.add_word(kw.lower(), kw.lower())
AC.make_automaton()

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # 轉小寫
    text = text.lower()
    # 分詞
    tokens = word_tokenize(text)
    # 英文還原
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # 組回句子
    return " ".join(tokens)

def match_ac(query, topn=80):
    # 預處理 query
    query_proc = preprocess_text(query)
    # 找出匹配到的 keyword
    keyword_scores = defaultdict(float)
    matched_keywords = [kw for _, kw in AC.iter(query_proc)]
    for kw in matched_keywords:
        keyword_scores[kw] += 1
    # 計算每個 keyword 對應的問題
    question_scores = defaultdict(float)
    for kw, score in keyword_scores.items():
        qids = builder.keyword_to_questions.get(kw, [])
        for qid in qids:
            question_scores[qid] += score
    # 排序取 topn
    ranked = sorted(question_scores.items(), key=lambda x: -x[1])
    return [qid for qid, _ in ranked[:topn]]

# === 檢查 AC coverage ===
results = []
for idx, row in query_df.iterrows():
    query = str(row["text"]).strip()
    true_label = row["true_label"]
    ac_top_qids = match_ac(query, topn=80)
    in_AC = int(true_label in ac_top_qids)
    results.append({
        "text": query,
        "true_label": true_label,
        "true_class": row["true_class"],
        "in_AC": in_AC
    })

out_df = pd.DataFrame(results)
out_df.to_excel(CURRENT_DIR / "test_with_AC_check.xlsx", index=False)
print("完成，已輸出 test_with_AC_check.xlsx")
