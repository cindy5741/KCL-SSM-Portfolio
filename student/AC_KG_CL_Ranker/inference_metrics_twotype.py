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
AC_TOPN = 50          # AC 檢索的候選數
CL_TOPN = 50          # 語意模型檢索的候選數
NEIGHBOR_WEIGHT = 0.1
GAMMA = 0.7
ALPHA = 0.9           # CL 模型分數權重 (1-ALPHA 為 AC 權重)
MU = 0.4
KEYWORD_THRESHOLD = 3
WEIGHT_STRATEGY = "small_increment"

SELECTED_KEYWORDS = []
SPECIAL_MAIN_WEIGHT = 0.0
SPECIAL_NEIGHBOR_WEIGHT = 0.0

# === 檔案路徑設定 (使用你上傳的 excel 格式) ===
MODEL_PATH = CURRENT_DIR.parent / "model" / "CL_student_2026-02-16-layer10"
KG_SOURCE_FILE = CURRENT_DIR / "第九版關鍵字.xlsx"
LABEL_MAPPING_FILE = CURRENT_DIR / "編號對應標準問句_增加類別251.xlsx"
TEST_DATA_FILE = CURRENT_DIR / "user_query.xlsx"

# === 1. 初始化模型與讀取資料 ===
print(f"Using device: {device}")
tokenizer_cl = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext", local_files_only=True)
cl_model = CLBERT.from_pretrained(str(MODEL_PATH), args={'config': "hfl/chinese-roberta-wwm-ext", 'dropout': 0.3, 'num_class': None}).to(device)
cl_model.eval()

# 讀取標準問句對照表
label_df = pd.read_excel(LABEL_MAPPING_FILE).dropna(subset=["知識編號", "標準問句"])
label_map = dict(zip(label_df["知識編號"].astype(float).astype(int), label_df["標準問句"]))
ALL_QIDS = list(label_map.keys())

# === 2. 雙向混合查詢系統 (AC + Graph) ===
class HybridQuerySystem:
    def __init__(self, builder, neighbor_weight, gamma, keyword_threshold, weight_strategy, selected_keywords, special_main, special_neighbor):
        self.builder = builder
        self.neighbor_weight = neighbor_weight
        self.gamma = gamma
        self.keyword_threshold = keyword_threshold
        self.weight_strategy = weight_strategy
        self.selected_keywords = selected_keywords
        self.special_main = special_main
        self.special_neighbor = special_neighbor
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
        elif self.weight_strategy == "small_increment": return 1 + position * 0.1
        elif self.weight_strategy == "linear": return 1 + position
        return 1

    def match(self, query):
        keyword_scores = defaultdict(float)
        query_proc = self.preprocess(query)
        matched_kws = [kw for _, kw in self.AC.iter(query_proc)]
        
        use_weighting = len(matched_kws) > self.keyword_threshold
        for i, kw in enumerate(matched_kws):
            keyword_scores[kw] += self.get_weight_for_position(i) if use_weighting else 1.0

        for kw in list(keyword_scores.keys()):
            if kw in self.builder.graph:
                for n in self.builder.graph.neighbors(kw):
                    keyword_scores[n] += self.neighbor_weight

        # Specific keywords logic
        for kw in self.selected_keywords:
            if kw in matched_kws:
                keyword_scores[kw] += self.special_main
            else:
                for m_kw in matched_kws:
                    if m_kw in self.builder.graph and kw in self.builder.graph.neighbors(m_kw):
                        keyword_scores[kw] += self.special_neighbor
                        break

        question_scores = defaultdict(float)
        for kw, score in keyword_scores.items():
            qids = self.builder.keyword_to_questions.get(kw, [])
            for qid in qids:
                question_scores[qid] += score

        for qid in question_scores:
            num = len(self.builder.standard_questions.get(qid, []))
            if num > 0: question_scores[qid] /= (num ** self.gamma)
            
        return sorted(question_scores.items(), key=lambda x: -x[1])

# === 3. 實驗流程與指標計算 ===
def run_metrics_experiment():
    # 建立 KG
    builder = KnowledgeGraphBuilder(mu=MU)
    try:
        builder.load_data_from_excel(KG_SOURCE_FILE)
    except AttributeError:
        builder.load_data_from_excel(KG_SOURCE_FILE)
    builder.build_graph()
    
    system = HybridQuerySystem(builder, NEIGHBOR_WEIGHT, GAMMA, KEYWORD_THRESHOLD, WEIGHT_STRATEGY, SELECTED_KEYWORDS, SPECIAL_MAIN_WEIGHT, SPECIAL_NEIGHBOR_WEIGHT)

    # 預編碼全體 Candidate Embeddings (供 Dense 檢索使用)
    print("Encoding all candidates for Dense Retrieval...")
    candidate_embs_list = []
    batch_size = 64
    for i in range(0, len(ALL_QIDS), batch_size):
        batch_ids = ALL_QIDS[i:i+batch_size]
        batch_texts = [label_map[qid] for qid in batch_ids]
        inputs = tokenizer_cl(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = cl_model(**inputs)
            embs = out[0] if isinstance(out, (tuple, list)) else out
            if embs.dim() == 3: embs = embs[:, 0, :]
            candidate_embs_list.append(embs.float())
            
    # 將所有向量合併為一個矩陣，加速 Cosine Similarity 計算
    cand_tensor = torch.cat(candidate_embs_list, dim=0)

    # 指標初始化
    query_df = pd.read_excel(TEST_DATA_FILE)
    topk_hits = {1:0, 3:0, 5:0}; mrr_list = []; ndcg_list = {1:[], 3:[], 5:[]}
    t_sum_A = 0; t_sum_B = 0; count = 0

    for _, row in tqdm(query_df.iterrows(), total=len(query_df), desc="Inference"):
        query_text = str(row.get("text", "")).strip()
        true_label = row.get("true_label")
        if pd.isna(true_label): continue
        
        t0 = time.perf_counter()
        
        # === 路徑 1: Dense 語意檢索 ===
        q_inp = tokenizer_cl([query_text], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            q_out = cl_model(**q_inp)
            q_emb = q_out[0] if isinstance(q_out, (tuple, list)) else q_out
            if q_emb.dim() == 3: q_emb = q_emb[:, 0, :]
            q_emb = q_emb.float()
        
        sims = torch.nn.functional.cosine_similarity(q_emb, cand_tensor).cpu().numpy()
        
        # 取得 CL 的 Top-N
        if CL_TOPN < len(sims):
            cl_top_idx = np.argpartition(-sims, CL_TOPN)[:CL_TOPN]
        else:
            cl_top_idx = np.arange(len(sims))
        cl_top_qids = [ALL_QIDS[i] for i in cl_top_idx]
        qid_to_sim = {ALL_QIDS[i]: float(sims[i]) for i in range(len(sims))}

        # === 路徑 2: Sparse 關鍵字檢索 ===
        matched_questions = system.match(query_text)
        ac_top_qids = [qid for qid, _ in matched_questions[:AC_TOPN]]
        ac_score_map = {qid: score for qid, score in matched_questions}

        # === 融合 (Union) ===
        candidate_qids = list(dict.fromkeys(cl_top_qids + ac_top_qids))
        
        if not candidate_qids:
            mrr_list.append(0); [ndcg_list[k].append(0) for k in [1, 3, 5]]; count += 1; continue

        # 提取分數並進行 Min-Max 正規化
        sim_vals = np.array([qid_to_sim[q] for q in candidate_qids], dtype=float)
        sim_min, sim_max = sim_vals.min(), sim_vals.max()
        sim_norm_map = {q: 0.5 if sim_max == sim_min else (qid_to_sim[q] - sim_min) / (sim_max - sim_min) for q in candidate_qids}

        kw_vals = np.array([ac_score_map.get(q, 0.0) for q in candidate_qids], dtype=float)
        kw_min, kw_max = kw_vals.min(), kw_vals.max()
        kw_norm_map = {q: 0.5 if kw_max == kw_min else (ac_score_map.get(q, 0.0) - kw_min) / (kw_max - kw_min) for q in candidate_qids}

        # 計算最終 Hybrid 分數
        score_map = {}
        for qid in candidate_qids:
            score_map[qid] = ALPHA * sim_norm_map[qid] + (1 - ALPHA) * kw_norm_map[qid]

        # 排序結果
        final_ranked = sorted(score_map.keys(), key=lambda x: -score_map[x])
        
        tA_end = time.perf_counter() # Latency A 結束點

        # === 模擬 Latency B: 即時動態編碼所有聯集候選句 ===
        union_texts = [label_map[qid] for qid in candidate_qids if qid in label_map]
        if union_texts:
            tk_inp = tokenizer_cl(union_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                cl_model(**tk_inp)
        tB_end = time.perf_counter() # Latency B 結束點

        # === 計算指標 ===
        try:
            target_id = int(float(true_label))
            if target_id in final_ranked:
                rank = final_ranked.index(target_id) + 1
                for k in [1, 3, 5]:
                    if rank <= k: topk_hits[k] += 1
                    ndcg_list[k].append(1.0 / np.log2(rank + 1) if rank <= k else 0)
                mrr_list.append(1.0 / rank)
            else:
                mrr_list.append(0); [ndcg_list[k].append(0) for k in [1, 3, 5]]
        except (ValueError, KeyError):
            mrr_list.append(0); [ndcg_list[k].append(0) for k in [1, 3, 5]]

        t_sum_A += (tA_end - t0); t_sum_B += (tB_end - t0); count += 1

    # 輸出最終報表
    print(f"\n=== 雙向 Hybrid 策略指標結果 (N={count}) ===")
    for k in [1, 3, 5]:
        acc = topk_hits[k]/count if count > 0 else 0
        ndcg = np.mean(ndcg_list[k]) if count > 0 else 0
        print(f"Top-{k} Accuracy: {acc:.4f} | NDCG@{k}: {ndcg:.4f}")
    
    mrr = np.mean(mrr_list) if count > 0 else 0
    print(f"MRR: {mrr:.4f}")
    
    print(f"\n=== 推論效率 (Latency) ===")
    print(f"A) 平均耗時 (使用離線編碼候選句): {(t_sum_A/count)*1000:.3f} ms/query")
    print(f"B) 平均耗時 (動態即時編碼聯集句): {(t_sum_B/count)*1000:.3f} ms/query")

if __name__ == "__main__":
    run_metrics_experiment()