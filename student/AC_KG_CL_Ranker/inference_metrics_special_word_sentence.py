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

# === 核心參數設定 (請依照 Grid Search 最佳值填入) ===
AC_TOPN = 50
NEIGHBOR_WEIGHT = 0.1
GAMMA = 0.7
RERANK_THRESHOLD = 0.7
ALPHA = 0.9
MU = 0.4
KEYWORD_THRESHOLD = 3
WEIGHT_STRATEGY = "small_increment"

# === 特定指定策略參數 ===
# SELECTED_KEYWORDS = []
SELECTED_KEYWORDS = ["申訴", "不合理", "不服", "有問題"]
SPECIAL_MAIN_WEIGHT = 1.0       # 命中特定關鍵字的加分
SPECIAL_NEIGHBOR_WEIGHT = 0.5   # 特定關鍵字鄰居的加分
SPECIAL_STANDQ_IDS = [1, 2]     # 觸發時強制加入的標準問句 ID
# SPECIAL_STANDQ_IDS = []     # 觸發時強制加入的標準問句 ID
SPECIAL_STANDQ_WEIGHT = 2.0     # 強制加入問句的額外加分

# === 檔案路徑設定 (延續使用你的 excel) ===
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

# === 2. 特定指定查詢系統 (Specific Query System) ===
class SpecificQuerySystem:
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
        special_keyword_hit = False
        
        use_weighting = len(matched_kws) > self.keyword_threshold
        for i, kw in enumerate(matched_kws):
            keyword_scores[kw] += self.get_weight_for_position(i) if use_weighting else 1.0

        for kw in list(keyword_scores.keys()):
            if kw in self.builder.graph:
                for n in self.builder.graph.neighbors(kw):
                    keyword_scores[n] += self.neighbor_weight

        # 核心：特定關鍵字加權邏輯
        for kw in self.selected_keywords:
            if kw in matched_kws:
                keyword_scores[kw] += self.special_main
                special_keyword_hit = True
            else:
                for mk in matched_kws:
                    if mk in self.builder.graph and kw in self.builder.graph.neighbors(mk):
                        keyword_scores[kw] += self.special_neighbor
                        special_keyword_hit = True
                        break

        question_scores = defaultdict(float)
        for kw, score in keyword_scores.items():
            qids = self.builder.keyword_to_questions.get(kw, [])
            for qid in qids:
                question_scores[qid] += score

        # 正規化
        for qid in question_scores:
            num = len(self.builder.standard_questions.get(qid, []))
            if num > 0: question_scores[qid] /= (num ** self.gamma)
            
        return sorted(question_scores.items(), key=lambda x: -x[1]), special_keyword_hit

# === 3. 實驗流程與指標計算 ===
def run_metrics_experiment():
    # 建立 KG
    builder = KnowledgeGraphBuilder(mu=MU)
    try:
        builder.load_data_from_excel(KG_SOURCE_FILE)
    except AttributeError:
        builder.load_data_from_excel(KG_SOURCE_FILE)
    builder.build_graph()

    system = SpecificQuerySystem(builder, NEIGHBOR_WEIGHT, GAMMA, KEYWORD_THRESHOLD, WEIGHT_STRATEGY, SELECTED_KEYWORDS, SPECIAL_MAIN_WEIGHT, SPECIAL_NEIGHBOR_WEIGHT)

    # 預編碼全體 Candidate Embeddings
    all_qids = list(label_map.keys())
    candidate_embs_dict = {qid: None for qid in all_qids}
    print("Encoding all candidates...")
    batch_size = 64
    for i in range(0, len(all_qids), batch_size):
        batch_ids = all_qids[i:i+batch_size]
        batch_texts = [label_map[qid] for qid in batch_ids]
        inputs = tokenizer_cl(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = cl_model(**inputs)
            embs = out[0] if isinstance(out, (tuple, list)) else out
            if embs.dim() == 3: embs = embs[:, 0, :]
            for qid, emb in zip(batch_ids, embs.float()):
                candidate_embs_dict[qid] = emb

    # 指標初始化
    query_df = pd.read_excel(TEST_DATA_FILE)
    topk_hits = {1:0, 3:0, 5:0}; mrr_list = []; ndcg_list = {1:[], 3:[], 5:[]}
    t_sum_A = 0; t_sum_B = 0; count = 0

    for _, row in tqdm(query_df.iterrows(), total=len(query_df), desc="Inference"):
        query_text = str(row.get("text", "")).strip()
        true_label = row.get("true_label")
        if pd.isna(true_label): continue
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        # 1. AC 粗篩與特定問句強制加入
        matched_questions, special_keyword_hit = system.match(query_text)
        matched_questions = list(matched_questions) # 轉成 list 方便後續修改
        ac_top_qids = [qid for qid, _ in matched_questions[:AC_TOPN]]
        
        # === 保底強制召回特定問句 ===
        if SPECIAL_STANDQ_IDS and special_keyword_hit:
            for sid in SPECIAL_STANDQ_IDS:
                if sid not in ac_top_qids and sid in label_map:
                    ac_top_qids.append(sid)
                    # 同步更新 matched_questions 分數 (為了後面的 Rerank 計算)
                    existing_sids = [q for q, s in matched_questions]
                    if sid not in existing_sids:
                        matched_questions.append((sid, 0.0))
                    for i, (qid, score) in enumerate(matched_questions):
                        if qid == sid:
                            matched_questions[i] = (qid, score + SPECIAL_STANDQ_WEIGHT)
                            break
        
        if not ac_top_qids:
            mrr_list.append(0); [ndcg_list[k].append(0) for k in [1, 3, 5]]; count += 1; continue

        # 2. Semantic Similarity
        q_inp = tokenizer_cl([query_text], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            q_out = cl_model(**q_inp)
            q_emb = q_out[0] if isinstance(q_out, (tuple, list)) else q_out
            if q_emb.dim() == 3: q_emb = q_emb[:, 0, :]
            q_emb = q_emb.float()
        
        cand_embs = torch.stack([candidate_embs_dict[qid] for qid in ac_top_qids])
        if device.type == "cuda":
            torch.cuda.synchronize()
        tA_end = time.perf_counter() # Latency A: 使用預先編碼

        # 模擬 Latency B: 動態即時編碼 Top-K 文本 (含強制加入的問句)
        topk_texts = [label_map[qid] for qid in ac_top_qids]
        tk_inp = tokenizer_cl(topk_texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        with torch.no_grad():
            cl_model(**tk_inp)

        if device.type == "cuda":
            torch.cuda.synchronize()
        tB_end = time.perf_counter()

        # Rerank 混合邏輯
        sims = torch.nn.functional.cosine_similarity(q_emb, cand_embs).cpu().numpy()
        qid_to_sim = {qid: sim for qid, sim in zip(ac_top_qids, sims)}
        
        final_ranked = []
        if max(sims) < RERANK_THRESHOLD:
            ac_map = dict(matched_questions)
            max_ac = max(ac_map.values()) if ac_map else 1.0
            norm_sims = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)
            combined = {qid: ALPHA * norm_sims[i] + (1-ALPHA) * (ac_map[qid]/max_ac) 
                        for i, qid in enumerate(ac_top_qids)}
            final_ranked = sorted(combined.keys(), key=lambda x: -combined[x])
        else:
            final_ranked = sorted(qid_to_sim.keys(), key=lambda x: -qid_to_sim[x])

        # 計算排序指標
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
    print(f"\n=== 指定策略指標結果 (N={count}) ===")
    for k in [1, 3, 5]:
        acc = topk_hits[k]/count if count > 0 else 0
        ndcg = np.mean(ndcg_list[k]) if count > 0 else 0
        print(f"Top-{k} Accuracy: {acc:.4f} | NDCG@{k}: {ndcg:.4f}")
    
    mrr = np.mean(mrr_list) if count > 0 else 0
    print(f"MRR: {mrr:.4f}")
    
    print(f"\n=== 推論效率 (Latency) ===")
    print(f"A) 平均耗時 (不含 TopK 重編碼): {(t_sum_A/count)*1000:.3f} ms/query")
    print(f"B) 平均耗時 (包含 TopK 重編碼): {(t_sum_B/count)*1000:.3f} ms/query")

if __name__ == "__main__":
    run_metrics_experiment()