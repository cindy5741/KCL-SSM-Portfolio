import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import ahocorasick
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer

# 確保 NLTK 資料集存在
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("下載 NLTK 資料集中...")
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("wordnet")

# === 路徑設定 ===
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR.parent))
sys.path.append(str(CURRENT_DIR.parent / "model"))

# 嘗試引用專案模組
try:
    from build_graph import KnowledgeGraphBuilder
except ImportError as e:
    print(f"匯入錯誤: {e}")
    print("請確認 build_graph.py 路徑是否正確")
    sys.exit(1)

warnings.filterwarnings('ignore')

# === 檔案設定 ===
KG_SOURCE_FILE = CURRENT_DIR / "第九版關鍵字.xlsx" 
TEST_DATA_FILE = CURRENT_DIR / "user_query.xlsx"

# === 參數設定 ===
AC_TOPN = 50        # 依據 many_excel 的列表
NEIGHBOR_WEIGHT = 0.1
GAMMA=0.7
MU = 0.4
KEYWORD_THRESHOLD = 3
WEIGHT_STRATEGY = "small_increment"

# === 載入資料 ===
print(f"Loading Label Map from: {KG_SOURCE_FILE.name}")
label_df = pd.read_excel(KG_SOURCE_FILE)

possible_id_cols = ["id", "qid", "question_id", "編號", "label", "category_id"]
possible_text_cols = ["question", "text", "標準問句", "center_sentence", "sentence", "content", "central_text"]

found_id_col = next((col for col in label_df.columns if str(col).lower() in possible_id_cols), None)
found_text_col = next((col for col in label_df.columns if str(col).lower() in possible_text_cols), None)

if not found_id_col:
    found_id_col = label_df.columns[0]
if not found_text_col:
    found_text_col = label_df.columns[1]

print(f"Mapping Columns -> ID: '{found_id_col}', Text: '{found_text_col}'")

label_df = label_df.dropna(subset=[found_id_col, found_text_col])
label_df["_id_str"] = label_df[found_id_col].astype(str).str.replace(r'\.0$', '', regex=True)
label_map = dict(zip(label_df["_id_str"], label_df[found_text_col]))

print(f"Loading Test Data from: {TEST_DATA_FILE.name}")
query_df = pd.read_excel(TEST_DATA_FILE)

possible_gt_cols = ["true_label", "label", "target", "ground_truth", "qid"]
found_gt_col = next((col for col in query_df.columns if str(col).lower() in possible_gt_cols), None)

if found_gt_col:
    query_df["true_label"] = query_df[found_gt_col].astype(str).str.replace(r'\.0$', '', regex=True)
else:
    print("警告: 測試資料無 Label 欄位。")
    query_df["true_label"] = None


# === QuerySystem ===
class QuerySystem:
    def __init__(self, builder: KnowledgeGraphBuilder, neighbor_weight=0.1, gamma=1.0,
                 keyword_threshold=5, weight_strategy="constant"):
        self.builder = builder
        self.neighbor_weight = neighbor_weight
        self.gamma = gamma
        self.keyword_threshold = keyword_threshold
        self.weight_strategy = weight_strategy
        self.lemmatizer = WordNetLemmatizer()

        self.AC = ahocorasick.Automaton()

        processed_to_original = defaultdict(set)
        for kw in self.builder.keyword_to_questions.keys():
            processed_kw = self.preprocess(kw)
            if processed_kw:
                processed_to_original[processed_kw].add(kw)

        for proc_kw, original_kws in processed_to_original.items():
            self.AC.add_word(proc_kw, original_kws)

        self.AC.make_automaton()

    def preprocess(self, text):
        if not isinstance(text, str):
            return ""
        return text.lower()

    def get_weight_for_position(self, position):
        if self.weight_strategy == "linear":
            return 1 + position
        elif self.weight_strategy == "increment":
            return 1 + position * 0.5
        elif self.weight_strategy == "small_increment":
            return 1 + position * 0.1
        else:
            return 1

    def get_keywords_with_weights(self, query):
        keyword_scores = defaultdict(float)

        query_proc = self.preprocess(str(query))

        matched_keywords = []
        for _, original_kws_set in self.AC.iter(query_proc):
            for kw in original_kws_set:
                matched_keywords.append(kw)

        use_weighting = len(matched_keywords) > self.keyword_threshold

        if use_weighting:
            for i, kw in enumerate(matched_keywords):
                keyword_scores[kw] += self.get_weight_for_position(i)
        else:
            for kw in matched_keywords:
                keyword_scores[kw] += 1

        current_keywords = list(keyword_scores.keys())
        for kw in current_keywords:
            if kw in self.builder.graph:
                for neighbor in self.builder.graph.neighbors(kw):
                    keyword_scores[neighbor] += self.neighbor_weight

        return keyword_scores, use_weighting

    def match_query_to_standard_question(self, query, normalize=True):
        kw_with_scores, use_weighting = self.get_keywords_with_weights(query)
        question_scores = defaultdict(float)

        for kw, score in kw_with_scores.items():
            qids = self.builder.keyword_to_questions.get(kw, [])
            for qid in qids:
                qid_str = str(qid).replace('.0', '')
                question_scores[qid_str] += score

        if normalize:
            for qid in question_scores:
                num = 0
                if qid in self.builder.standard_questions:
                    num = len(self.builder.standard_questions[qid])
                elif qid.isdigit() and int(qid) in self.builder.standard_questions:
                    num = len(self.builder.standard_questions[int(qid)])

                if num > 0:
                    question_scores[qid] /= (num ** self.gamma)

        return sorted(question_scores.items(), key=lambda x: -x[1]), use_weighting


# === 實驗流程：AC only ===
def run_experiment_ac_only():
    print("Building Knowledge Graph...")
    builder = KnowledgeGraphBuilder(mu=MU)
    builder.load_data_from_excel(KG_SOURCE_FILE)
    builder.build_graph()

    try:
        print(f"[Graph] nodes = {builder.graph.number_of_nodes():,} | edges = {builder.graph.number_of_edges():,}")
    except Exception as e:
        print(f"[Graph] cannot read graph stats: {e}")

    system = QuerySystem(
        builder,
        neighbor_weight=NEIGHBOR_WEIGHT,
        gamma=GAMMA,
        keyword_threshold=KEYWORD_THRESHOLD,
        weight_strategy=WEIGHT_STRATEGY
    )

    topk_hits = {1: 0, 3: 0, 5: 0}
    mrr_list = []
    ndcg1_list = []
    ndcg3_list = []
    ndcg5_list = []
    results = []

    def ndcg_from_rank(rank, k):
        if rank is None:
            return 0.0
        return 1.0 / np.log2(rank + 1) if rank <= k else 0.0

    print("Start Inference (AC only)...")

    t_sum = 0.0
    t_cnt = 0

    for idx, row in tqdm(query_df.iterrows(), total=len(query_df), ncols=100):
        query_text = row.get("text", row.get("query", row.get("question", "")))
        query_text = str(query_text).strip()
        true_label = row.get("true_label", None)

        if not true_label or true_label == "nan":
            continue

        t0 = time.perf_counter()

        matched_questions, _ = system.match_query_to_standard_question(query_text)

        # 只用 AC 分數排序
        ranked_items = matched_questions[:AC_TOPN]
        ranked_labels = [qid for qid, score in ranked_items]

        t1 = time.perf_counter()
        t_sum += (t1 - t0)
        t_cnt += 1

        if ranked_labels:
            try:
                rank_index = ranked_labels.index(true_label)
                rank = rank_index + 1
            except ValueError:
                rank_index = None
                rank = None

            top1_label = ranked_labels[0]
            top1_text = label_map.get(top1_label, "")
            top1_score = ranked_items[0][1]

            top3_labels = ranked_labels[:3]
            top3_texts = [label_map.get(qid, "") for qid in top3_labels]
            top3_scores = [score for _, score in ranked_items[:3]]

            top5_labels = ranked_labels[:5]
            top5_texts = [label_map.get(qid, "") for qid in top5_labels]
            top5_scores = [score for _, score in ranked_items[:5]]
        else:
            rank_index = None
            rank = None
            top1_label = None
            top1_text = ""
            top1_score = None
            top3_labels, top3_texts, top3_scores = [], [], []
            top5_labels, top5_texts, top5_scores = [], [], []

        for k in topk_hits.keys():
            if rank_index is not None and rank_index < k:
                topk_hits[k] += 1

        if rank_index is not None:
            mrr_list.append(1.0 / (rank_index + 1))
        else:
            mrr_list.append(0.0)

        ndcg1_list.append(ndcg_from_rank(rank, 1))
        ndcg3_list.append(ndcg_from_rank(rank, 3))
        ndcg5_list.append(ndcg_from_rank(rank, 5))

        results.append({
            "text": query_text,
            "true_label": true_label,
            "true_label_text": label_map.get(true_label, ""),
            "pred_top1_label": top1_label,
            "pred_top1_text": top1_text,
            "pred_top1_score": top1_score,
            "top3_labels": str(top3_labels),
            "top3_texts": str(top3_texts),
            "top3_scores": str([round(float(x), 6) for x in top3_scores]),
            "top5_labels": str(top5_labels),
            "top5_texts": str(top5_texts),
            "top5_scores": str([round(float(x), 6) for x in top5_scores]),
            "top1_correct": int(top1_label == true_label) if top1_label is not None else 0,
            "rank_of_true_label": rank,
            "reciprocal_rank": 1.0 / rank if rank is not None else 0.0
        })

    total = len(ndcg1_list)
    if total == 0:
        print("沒有有效的測試樣本。")
        return

    print("\n=== AC Only 指標結果 ===")
    print(f"Top-1: {topk_hits[1]/total:.4f}")
    print(f"Top-3: {topk_hits[3]/total:.4f}")
    print(f"Top-5: {topk_hits[5]/total:.4f}")
    print(f"MRR:   {np.mean(mrr_list):.4f}")
    print(f"NDCG@1: {np.mean(ndcg1_list):.4f}")
    print(f"NDCG@3: {np.mean(ndcg3_list):.4f}")
    print(f"NDCG@5: {np.mean(ndcg5_list):.4f}")

    if t_cnt > 0:
        mean_ms = (t_sum / t_cnt) * 1000.0
        print(f"\n=== 推論時間 (AC only) ===")
        print(f"{mean_ms:.3f} ms/query  (N={t_cnt})")

    results_df = pd.DataFrame(results)
    output_file = CURRENT_DIR / "prediction_results_ac_only.xlsx"
    results_df.to_excel(output_file, index=False, engine="openpyxl")
    print(f"\n已輸出每筆預測結果：{output_file}")


if __name__ == "__main__":
    run_experiment_ac_only()