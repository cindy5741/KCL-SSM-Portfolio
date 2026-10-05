import os
import sys
import time
import torch
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, logging
import ahocorasick
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# === NLTK 資料檢查 ===
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download("punkt")
    nltk.download("wordnet")

# === 路徑與模型設定 ===
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR.parent))
sys.path.append(str(CURRENT_DIR.parent / "model"))

from build_graph import KnowledgeGraphBuilder
from model import CLBERT

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 核心參數設定 (根據你的 Grid Search 最佳值調整) ===
AC_TOPN = 50
NEIGHBOR_WEIGHT = 1
GAMMA = 0.7
RERANK_THRESHOLD = 0.7
ALPHA = 0.9
MU = 0.4
KEYWORD_THRESHOLD = 3
WEIGHT_STRATEGY = "small_increment"
CATEGORY_BONUS = 1.0  # 階層加分權重

# 特定關鍵字加權設定
SELECTED_KEYWORDS = []
SPECIAL_MAIN_WEIGHT = 0.0
SPECIAL_NEIGHBOR_WEIGHT = 0.0

MODEL_PATH = CURRENT_DIR.parent / "model" / "CL_student_2026-02-16-layer10"
KG_SOURCE_FILE = CURRENT_DIR / "第九版關鍵字.xlsx"
CAT_MAPPING_FILE = CURRENT_DIR / "類別對應標準問句.xlsx"
TEST_DATA_FILE = CURRENT_DIR / "user_query.xlsx"

# === 1. 初始化模型與讀取資料 ===
tokenizer_cl = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext", local_files_only=True)
cl_model = CLBERT.from_pretrained(str(MODEL_PATH), args={'config': "hfl/chinese-roberta-wwm-ext", 'dropout': 0.3, 'num_class': None}).to(device)
cl_model.eval()

# 讀取標準問句對照表
label_df = pd.read_excel(CURRENT_DIR / "編號對應標準問句_增加類別251.xlsx").dropna(subset=["知識編號", "標準問句"])
label_map = dict(zip(label_df["知識編號"].astype(int), label_df["標準問句"]))

# === 2. 階層式查詢系統 (整合 NLTK 與 Category Bonus) ===
class HierarchyQuerySystem:
    def __init__(self, builder, neighbor_weight, gamma, keyword_threshold, weight_strategy, 
                 category_keywords, category_bonus):
        self.builder = builder
        self.neighbor_weight = neighbor_weight
        self.gamma = gamma
        self.keyword_threshold = keyword_threshold
        self.weight_strategy = weight_strategy
        self.category_keywords = category_keywords
        self.category_bonus = category_bonus
        self.lemmatizer = WordNetLemmatizer()
        
        self.AC = ahocorasick.Automaton()
        for kw in self.builder.keyword_to_questions.keys():
            self.AC.add_word(kw, kw)
        self.AC.make_automaton()

    def preprocess(self, text):
        if not isinstance(text, str):
            return ""
        return text.lower()

    def get_weight_for_position(self, position):
        if self.weight_strategy == "increment": return 1 + position * 0.5
        elif self.weight_strategy == "linear": return 1 + position
        elif self.weight_strategy == "small_increment": return 1 + position * 0.1
        return 1

    def match(self, query):
        keyword_scores = defaultdict(float)
        query_proc = self.preprocess(query)
        matched_kws = [kw for _, kw in self.AC.iter(query_proc)]
        
        # 基礎 AC 權重
        use_weighting = len(matched_kws) > self.keyword_threshold
        for i, kw in enumerate(matched_kws):
            keyword_scores[kw] += self.get_weight_for_position(i) if use_weighting else 1.0

        # 圖擴散 (Graph Diffusion)
        for kw in list(keyword_scores.keys()):
            if kw in self.builder.graph:
                for n in self.builder.graph.neighbors(kw):
                    keyword_scores[n] += self.neighbor_weight

        # 核心：類別加分邏輯
        question_scores = defaultdict(float)
        for kw, score in keyword_scores.items():
            qids = self.builder.keyword_to_questions.get(kw, [])
            is_category = kw in self.category_keywords
            for qid in qids:
                question_scores[qid] += score
                if is_category: # 若命中類別關鍵字，該類別下問句全加分
                    question_scores[qid] += self.category_bonus

        # 正規化
        for qid in question_scores:
            num = len(self.builder.standard_questions.get(qid, []))
            if num > 0: question_scores[qid] /= (num ** self.gamma)
            
        return sorted(question_scores.items(), key=lambda x: -x[1])

# === 3. 實驗流程與指標計算 ===
def run_metrics_experiment():
    # 建立 KG 並注入階層類別
    builder = KnowledgeGraphBuilder(mu=MU)
    builder.load_data_from_excel(KG_SOURCE_FILE)
    builder.build_graph()
    
    cat_kws = set()
    if CAT_MAPPING_FILE.exists():
        cat_df = pd.read_excel(CAT_MAPPING_FILE)
        for _, row in cat_df.iterrows():
            cat_kw = str(row['主題']).strip()
            cat_kws.add(cat_kw)
            qid = int(row['編號'])
            if cat_kw not in builder.keyword_to_questions:
                builder.keyword_to_questions[cat_kw] = set()
            builder.keyword_to_questions[cat_kw].add(qid)

    system = HierarchyQuerySystem(builder, NEIGHBOR_WEIGHT, GAMMA, KEYWORD_THRESHOLD, WEIGHT_STRATEGY, cat_kws, CATEGORY_BONUS)

    # 預編碼全體 Candidate Embeddings
    all_qids = list(label_map.keys())
    candidate_embs_dict = {qid: None for qid in all_qids}
    print("Encoding all candidates...")
    for i in range(0, len(all_qids), 64):
        batch_ids = all_qids[i:i+64]
        batch_texts = [label_map[qid] for qid in batch_ids]
        inputs = tokenizer_cl(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = cl_model(**inputs)
            # === 安全取出維度修正 ===
            embs = out[0] if isinstance(out, (tuple, list)) else out
            if embs.dim() == 3:
                embs = embs[:, 0, :]
            embs = embs.float()
            
            for qid, emb in zip(batch_ids, embs):
                candidate_embs_dict[qid] = emb

    # 指標初始化
    query_df = pd.read_excel(TEST_DATA_FILE)
    topk_hits = {1:0, 3:0, 5:0}; mrr_list = []; ndcg_list = {1:[], 3:[], 5:[]}
    t_sum_A = 0; t_sum_B = 0; count = 0

    for _, row in tqdm(query_df.iterrows(), total=len(query_df), desc="Inference"):
        query_text = str(row.get("text", "")).strip()
        true_label = row.get("true_label")
        if pd.isna(true_label): continue
        
        t0 = time.perf_counter()
        # 1. AC + Hierarchy Match
        matched = system.match(query_text)
        top_qids = [qid for qid, _ in matched[:AC_TOPN]]
        
        if not top_qids:
            mrr_list.append(0); count += 1; continue

        # 2. Semantic Similarity
        q_inp = tokenizer_cl([query_text], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            q_out = cl_model(**q_inp)
            # === 安全取出維度修正 ===
            q_emb = q_out[0] if isinstance(q_out, (tuple, list)) else q_out
            if q_emb.dim() == 3:
                q_emb = q_emb[:, 0, :]
            q_emb = q_emb.float()
        
        cand_embs = torch.stack([candidate_embs_dict[qid] for qid in top_qids])
        tA_end = time.perf_counter() # 時間 A：不含 TopK 重新編碼

        # 模擬 B：包含 TopK 重新編碼 (針對這 N 條再跑一次 Encoder)
        topk_texts = [label_map[qid] for qid in top_qids]
        tk_inp = tokenizer_cl(topk_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            cl_model(**tk_inp)
        tB_end = time.perf_counter()

        # Rerank 邏輯
        sims = torch.nn.functional.cosine_similarity(q_emb, cand_embs).cpu().numpy()
        qid_to_sim = {qid: sim for qid, sim in zip(top_qids, sims)}
        
        final_ranked = []
        if max(sims) < RERANK_THRESHOLD:
            # 混合 AC 分數與 CL 分數
            ac_map = dict(matched)
            max_ac = max(ac_map.values()) if ac_map else 1.0
            norm_sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)
            combined = {qid: ALPHA * norm_sims[i] + (1-ALPHA) * (ac_map[qid]/max_ac) 
                        for i, qid in enumerate(top_qids)}
            final_ranked = sorted(combined.keys(), key=lambda x: -combined[x])
        else:
            final_ranked = sorted(qid_to_sim.keys(), key=lambda x: -qid_to_sim[x])

        # 計算指標
        try:
            rank = final_ranked.index(int(true_label)) + 1
            for k in [1, 3, 5]:
                if rank <= k: topk_hits[k] += 1
                ndcg_list[k].append(1.0 / np.log2(rank + 1) if rank <= k else 0)
            mrr_list.append(1.0 / rank)
        except (ValueError, KeyError):
            mrr_list.append(0); [ndcg_list[k].append(0) for k in [1, 3, 5]]

        t_sum_A += (tA_end - t0); t_sum_B += (tB_end - t0); count += 1

    # 輸出結果
    print(f"\n=== 階層式策略指標 (N={count}) ===")
    for k in [1, 3, 5]: print(f"Top-{k}: {topk_hits[k]/count:.4f} | NDCG@{k}: {np.mean(ndcg_list[k]):.4f}")
    print(f"MRR: {np.mean(mrr_list):.4f}")
    print(f"\n=== 推論耗時 (Latency) ===")
    print(f"A) 不含 TopK 重新編碼: {(t_sum_A/count)*1000:.3f} ms/query")
    print(f"B) 包含 TopK 重新編碼: {(t_sum_B/count)*1000:.3f} ms/query")

if __name__ == "__main__":
    run_metrics_experiment()