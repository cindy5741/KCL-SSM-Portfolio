import os
import sys
import time
import torch
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer
import ahocorasick
import nltk
from nltk.stem import WordNetLemmatizer

# === 路徑設定 ===
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR.parent))
sys.path.append(str(CURRENT_DIR.parent / "model"))

try:
    from build_graph import KnowledgeGraphBuilder
    from model import CLBERT
except ImportError as e:
    print(f"匯入錯誤: {e}。請確認路徑。")
    sys.exit(1)

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 基礎參數設定 ===
KG_SOURCE_FILE = CURRENT_DIR / "第九版關鍵字.xlsx" 
CONFIG = "hfl/chinese-roberta-wwm-ext"
DROPOUT = 0.3
MODEL_PATH = CURRENT_DIR.parent / "model" / "CL_student_2026-04-27-lamda09"

# 最佳化參數 (Grid Search 結果)
AC_TOPN = 50        
NEIGHBOR_WEIGHT = 0.1
GAMMA = 0.7
RERANK_THRESHOLD = 0.7 
ALPHA = 0.9         
MU = 0.4
KEYWORD_THRESHOLD = 3
WEIGHT_STRATEGY = "small_increment"

# === 移植 QuerySystem 類別 ===
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
        if not isinstance(text, str): return ""
        return text.lower()

    def get_weight_for_position(self, position):
        if self.weight_strategy == "linear": return 1 + position
        elif self.weight_strategy == "increment": return 1 + position * 0.5
        elif self.weight_strategy == "small_increment": return 1 + position * 0.1
        else: return 1

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

# === 預載入與初始化系統 ===
print("=== 正在初始化單句預測系統 ===")
print("Using device:", device)

# 載入模型與分詞器
init_params = {'config': CONFIG, 'dropout': DROPOUT, 'num_class': None}
tokenizer_cl = AutoTokenizer.from_pretrained(init_params['config'], local_files_only=True)
cl_model = CLBERT.from_pretrained(str(MODEL_PATH), args=init_params, local_files_only=True).to(device)
cl_model.eval()

# 建立知識圖譜
builder = KnowledgeGraphBuilder(mu=MU)
builder.load_data_from_excel(KG_SOURCE_FILE)
builder.build_graph()
system = QuerySystem(builder, neighbor_weight=NEIGHBOR_WEIGHT, gamma=GAMMA,
                    keyword_threshold=KEYWORD_THRESHOLD, weight_strategy=WEIGHT_STRATEGY)

# 讀取標準問句對照表
label_df = pd.read_excel(KG_SOURCE_FILE)
possible_id_cols = ["id", "qid", "question_id", "編號", "label", "category_id"]
possible_text_cols = ["question", "text", "標準問句", "center_sentence", "sentence", "content", "central_text"]
found_id_col = next((col for col in label_df.columns if str(col).lower() in possible_id_cols), label_df.columns[0])
found_text_col = next((col for col in label_df.columns if str(col).lower() in possible_text_cols), label_df.columns[1])

label_df = label_df.dropna(subset=[found_id_col, found_text_col])
label_df["_id_str"] = label_df[found_id_col].astype(str).str.replace(r'\.0$', '', regex=True)
label_map = dict(zip(label_df["_id_str"], label_df[found_text_col]))

# 預計算所有 Candidates 向量 (模擬離線環境)
print("正在預計算所有標準問句的 Embeddings (Offline)...")
all_label_ids = list(label_map.keys())
all_label_texts = [label_map[qid] for qid in all_label_ids]
candidate_embs_dict = {}

batch_size = 64
for i in range(0, len(all_label_texts), batch_size):
    batch_texts = all_label_texts[i:i+batch_size]
    batch_ids = all_label_ids[i:i+batch_size]
    batch_inputs = tokenizer_cl(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        batch_embs = cl_model(**batch_inputs)
        if isinstance(batch_embs, (tuple, list)): batch_embs = batch_embs[0]
        if batch_embs.dim() == 3: batch_embs = batch_embs[:, 0, :]
        batch_embs = batch_embs.float()
    for qid, emb in zip(batch_ids, batch_embs):
        candidate_embs_dict[qid] = emb

print("=== 系統初始化完成，可以開始預測 ===\n")

# === 單句預測核心函式 ===
def predict_single_query(query_text: str):
    query_text = str(query_text).strip()
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # 1. AC + Graph 篩選
    matched_questions, _ = system.match_query_to_standard_question(query_text)
    ac_top_qids = [qid for qid, _ in matched_questions[:AC_TOPN]]
    
    if not ac_top_qids:
        return {"error": "AC 自動機未匹配到任何候選關鍵字"}

    # 2. Query 進 CL Encoder 得到向量
    q_inputs = tokenizer_cl([query_text], return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
    with torch.no_grad():
        q_emb = cl_model(**q_inputs)
        if isinstance(q_emb, (tuple, list)): q_emb = q_emb[0]
        if q_emb.dim() == 3: q_emb = q_emb[:, 0, :]
        q_emb = q_emb.float()

    # 從預計算字典取出 Top-K 候選向量
    cand_embs = torch.stack([candidate_embs_dict[qid] for qid in ac_top_qids], dim=0)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    tA_end = time.perf_counter()  # ---- Time A 截止點 ----

    # 3. 模擬「實時編碼」：把 Top-K 的文字重新再進一次 Encoder 計時
    topk_texts = [label_map[qid] for qid in ac_top_qids if qid in label_map]
    if len(topk_texts) > 0:
        topk_inputs = tokenizer_cl(topk_texts, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        with torch.no_grad():
            topk_out = cl_model(**topk_inputs)
            if isinstance(topk_out, (tuple, list)): topk_out = topk_out[0]
            if topk_out.dim() == 3: topk_out = topk_out[:, 0, :]
            _ = topk_out.float()

    if device.type == "cuda":
        torch.cuda.synchronize()
    tB_end = time.perf_counter()  # ---- Time B 截止點 ----

    # 4. 計算餘弦相似度與 Rerank 評分
    q_emb_n = torch.nn.functional.normalize(q_emb, p=2, dim=1)
    c_embs_n = torch.nn.functional.normalize(cand_embs, p=2, dim=1)
    sims = torch.matmul(q_emb_n, c_embs_n.T)[0].cpu().numpy()
    qid_to_sim = {qid: sim for qid, sim in zip(ac_top_qids, sims)}
    
    pred_label_cl = max(qid_to_sim, key=qid_to_sim.get)
    top_score_cl = qid_to_sim[pred_label_cl]
    
    final_scores = {}
    if top_score_cl < RERANK_THRESHOLD:
        sims_norm_vals = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)
        qid_to_sim_norm = {qid: val for qid, val in zip(ac_top_qids, sims_norm_vals)}
        
        ac_score_dict = {qid: score for qid, score in matched_questions if qid in ac_top_qids}
        max_ac_score = max(ac_score_dict.values()) if ac_score_dict else 1.0
        ac_score_normalized = {qid: score / max_ac_score for qid, score in ac_score_dict.items()}

        for qid in ac_top_qids:
            final_scores[qid] = ALPHA * qid_to_sim_norm[qid] + (1 - ALPHA) * ac_score_normalized.get(qid, 0)
    else:
        final_scores = qid_to_sim
        
    ranked_labels = sorted(final_scores.keys(), key=lambda x: -final_scores[x])
    
    # 計算單次耗時 (毫秒)
    latency_ms_A = (tA_end - t0) * 1000.0
    latency_ms_B = (tB_end - t0) * 1000.0

    # 回傳 Top-5 預測結果與時間
    output = {
        "query": query_text,
        "latency_A_ms": latency_ms_A,
        "latency_B_ms": latency_ms_B,
        "predictions": []
    }
    
    for rank, qid in enumerate(ranked_labels[:5], 1):
        output["predictions"].append({
            "rank": rank,
            "qid": qid,
            "standard_question": label_map.get(qid, "未知"),
            "score": round(float(final_scores[qid]), 4)
        })
    return output

if __name__ == "__main__":
    # 測試單句輸入
    test_query = "我申訴去哪裡。"
    res = predict_single_query(test_query)
    
    if "error" in res:
        print(res["error"])
    else:
        print(f"輸入句子: {res['query']}")
        print("-" * 50)
        print(f"A) 不含 topK centers 進 encoder 的時間: {res['latency_A_ms']:.3f} ms")
        print(f"B) 含 topK centers 進 encoder 的時間:  {res['latency_B_ms']:.3f} ms")
        print("-" * 50)
        print("Top-5 預測匹配結果:")
        for pred in res["predictions"]:
            print(f" Rank {pred['rank']}: [{pred['qid']}] {pred['standard_question']} (Score: {pred['score']})")