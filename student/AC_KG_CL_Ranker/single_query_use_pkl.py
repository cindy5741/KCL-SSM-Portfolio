import os
import sys
import time
import torch
import warnings
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, logging
import ahocorasick

# === 路徑設定 ===
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "model"))

from model import CLBERT
from model import CLBERT

# === 固定參數 ===
CONFIG = "hfl/chinese-roberta-wwm-ext"
DROPOUT = 0.3
NUM_CLASS = 142655
MODEL_PATH = ROOT_DIR / "model" / "CL_2025-09-07"

logging.set_verbosity_error()
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 初始化模型 ===
init_params = {
    'config': CONFIG,
    'dropout': DROPOUT,
    'num_class': NUM_CLASS,
}
tokenizer_cl = AutoTokenizer.from_pretrained(init_params['config'], local_files_only=True)
cl_model = CLBERT.from_pretrained(str(MODEL_PATH), args=init_params, local_files_only=True).to(device)
cl_model.eval()

# === 載入資料 ===
label_df = pd.read_excel(CURRENT_DIR / "編號對應標準問句.xlsx")
label_df = label_df.dropna(subset=["知識編號", "標準問句"])
label_df["知識編號"] = label_df["知識編號"].astype(int)
label_map = dict(zip(label_df["知識編號"], label_df["標準問句"]))

query_df = pd.read_excel(CURRENT_DIR / "user_query.xlsx")

datetime_str = time.strftime('%Y%m%d_%H%M%S')
OUT_DETAIL_DIR = CURRENT_DIR / f"single_query_result_{datetime_str}"
OUT_DETAIL_DIR.mkdir(exist_ok=True)

# === QuerySystem 改版，支援加權策略 ===
class QuerySystem:
    def __init__(self, graph, keyword_to_questions, neighbor_weight=0.1, gamma=1.0,
                 keyword_threshold=5, weight_strategy="constant"):
        self.graph = graph
        self.keyword_to_questions = keyword_to_questions
        self.neighbor_weight = neighbor_weight
        self.gamma = gamma
        self.keyword_threshold = keyword_threshold
        self.weight_strategy = weight_strategy
        self.AC = ahocorasick.Automaton()
        for kw in self.keyword_to_questions.keys():
            self.AC.add_word(kw, kw)
        self.AC.make_automaton()

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
        matched_keywords = []
        for _, kw in self.AC.iter(query):
            matched_keywords.append(kw)

        use_weighting = len(matched_keywords) > self.keyword_threshold
        if use_weighting:
            for i, kw in enumerate(matched_keywords):
                keyword_scores[kw] += self.get_weight_for_position(i)
        else:
            for kw in matched_keywords:
                keyword_scores[kw] += 1

        for kw in list(keyword_scores.keys()):
            if kw in self.graph:
                for neighbor in self.graph.neighbors(kw):
                    keyword_scores[neighbor] += self.neighbor_weight

        return keyword_scores, use_weighting

    def match_query_to_standard_question(self, query, normalize=True, gamma=1.0):
        kw_with_scores, use_weighting = self.get_keywords_with_weights(query)
        question_scores = defaultdict(float)
        for kw, score in kw_with_scores.items():
            qids = self.keyword_to_questions.get(kw, [])
            for qid in qids:
                question_scores[qid] += score
        if normalize:
            for qid in question_scores:
                num_keywords = len(self.keyword_to_questions.get(qid, []))
                if num_keywords > 0:
                    question_scores[qid] /= (num_keywords ** gamma)
        return sorted(question_scores.items(), key=lambda x: -x[1]), use_weighting


# === 單句查詢 ===
def run_single_query(query_sentence, neighbor_weight, ac_topn,
                     rerank_threshold, alpha, keyword_threshold, weight_strategy):

    kg_path = CURRENT_DIR / "knowledge_graph.pkl"

    if not kg_path.exists():
        print("⚠️ 找不到 knowledge_graph.pkl，請先生成它。")
        return

    with open(kg_path, "rb") as f:
        kg_data = pickle.load(f)

    graph = kg_data["graph"]
    keyword_to_questions = kg_data["keyword_to_questions"]

    system = QuerySystem(graph,
                         keyword_to_questions,
                         neighbor_weight=neighbor_weight,
                         gamma=1.0,
                         keyword_threshold=keyword_threshold,
                         weight_strategy=weight_strategy)

    start_time = time.time()
    matched_questions, use_weighting = system.match_query_to_standard_question(query_sentence, normalize=True)
    ac_ranked = [qid for qid, _ in matched_questions]

    if not ac_ranked:
        print("❌ 沒有匹配到任何標準問句")
        return

    ac_top_qids = ac_ranked[:ac_topn]

    with torch.no_grad():
        query_inputs = tokenizer_cl([query_sentence], return_tensors="pt", padding=True, truncation=True).to(device)
        query_emb = cl_model(**query_inputs)
        if isinstance(query_emb, (tuple, list)):
            query_emb = query_emb[0]
        if query_emb.dim() == 3:
            query_emb = query_emb[:, 0, :]
        elif query_emb.dim() == 2:
            pass

        candidate_texts = [label_map[qid] for qid in ac_top_qids]
        candidate_inputs = tokenizer_cl(candidate_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        candidate_embs = cl_model(**candidate_inputs)
        if isinstance(candidate_embs, (tuple, list)):
            candidate_embs = candidate_embs[0]
        if candidate_embs.dim() == 3:
            candidate_embs = candidate_embs[:, 0, :]
        elif candidate_embs.dim() == 2:
            pass

    sims = cosine_similarity(query_emb.cpu().numpy(), candidate_embs.cpu().numpy())[0]
    qid_to_sim = {qid: sim for qid, sim in zip(ac_top_qids, sims)}
    pred_label_cl = max(qid_to_sim, key=qid_to_sim.get)
    top_score_cl = qid_to_sim[pred_label_cl]

    pred_label_final = pred_label_cl
    top_score_final = top_score_cl
    rerank_flag = 0

    if top_score_cl < rerank_threshold:
        rerank_flag = 1
        sims_array = np.array(sims)
        sims_norm = (sims_array - sims_array.min()) / (sims_array.max() - sims_array.min() + 1e-8)
        qid_to_sim_norm = {qid: sim for qid, sim in zip(ac_top_qids, sims_norm)}

        ac_score_dict = {qid: score for qid, score in matched_questions if qid in ac_top_qids}
        max_ac_score = max(ac_score_dict.values()) if ac_score_dict else 1.0
        ac_score_normalized = {qid: score / max_ac_score for qid, score in ac_score_dict.items()}

        combined_scores = {
            qid: alpha * qid_to_sim_norm[qid] + (1 - alpha) * ac_score_normalized[qid]
            for qid in ac_top_qids
        }

        pred_label_final = max(combined_scores, key=combined_scores.get)
        top_score_final = combined_scores[pred_label_final]

    print("\n=== 查詢結果 ===")
    print(f"查詢句: {query_sentence}")
    print(f"是否用加權策略: {int(use_weighting)}, 加權策略: {weight_strategy if use_weighting else 'constant'}")
    print(f"最終預測: {label_map.get(pred_label_final, '未知')} (分數: {top_score_final:.4f})")
    print(f"CL過濾 TOP1: {label_map.get(pred_label_cl, '未知')} (分數: {top_score_cl:.4f})")
    print(f"是否用 rerank: {rerank_flag}")
    print(f"耗時: {time.time() - start_time:.2f} 秒")

    print("\n=== 250句排名 ===")
    results = []
    for rank, qid in enumerate(ac_ranked, start=1):
        score = dict(matched_questions)[qid]
        print(f"Rank {rank}: QID={qid}, 分數={score:.4f}, 標準問句={label_map.get(qid, '未知')}")
        results.append({
            "Rank": rank,
            "QID": qid,
            "Score": score,
            "Standard Question": label_map.get(qid, "未知")
        })

    # 存成 Excel
    single_results = pd.DataFrame(results)
    output_file = OUT_DETAIL_DIR / f"single_query_result.xlsx"
    single_results.to_excel(output_file, index=False, encoding="utf-8")
    print(f"\n✅ 結果已儲存到 {output_file}")
    return matched_questions


if __name__ == "__main__":
    query_sentence = "我要查詢交通違規的罰單"
    neighbor_weight = 0.5
    ac_topn = 80
    rerank_threshold = 0.5
    alpha = 0.5
    keyword_threshold = 5
    weight_strategy = "increment"

    run_single_query(query_sentence, neighbor_weight, ac_topn,
                     rerank_threshold, alpha, keyword_threshold, weight_strategy)
