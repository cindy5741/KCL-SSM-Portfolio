# import os
# import sys
# import time
# import torch
# import warnings
# import pandas as pd
# import numpy as np
# from pathlib import Path
# from collections import defaultdict
# from itertools import product
# from transformers import AutoTokenizer, logging
# import ahocorasick
# from tqdm import tqdm

# # === 路徑設定 ===
# CURRENT_DIR = Path(__file__).resolve().parent
# sys.path.append(str(CURRENT_DIR.parent))
# sys.path.append(str(CURRENT_DIR.parent / "model"))

# from build_graph import KnowledgeGraphBuilder
# from model import CLBERT

# warnings.filterwarnings('ignore')
# logging.set_verbosity_error()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# # === 固定參數 (只跑一組) ===
# AC_TOPN = 40
# NEIGHBOR_WEIGHT = 0.1
# GAMMA=0.1
# RERANK_THRESHOLD = 0.7
# ALPHA = 0.5
# MU = 0.4
# KEYWORD_THRESHOLD = 3
# WEIGHT_STRATEGY = "increment"
# CONFIG = "hfl/chinese-roberta-wwm-ext"
# DROPOUT = 0.3
# MODEL_PATH = CURRENT_DIR.parent / "model" / "CL_student_2026-01-18"

# # === 初始化模型 ===
# init_params = {'config': CONFIG, 'dropout': DROPOUT, 'num_class': None}
# tokenizer_cl = AutoTokenizer.from_pretrained(init_params['config'], local_files_only=True)
# cl_model = CLBERT.from_pretrained(str(MODEL_PATH), args=init_params, local_files_only=True).to(device)
# cl_model.eval()

# # === 載入資料 ===
# label_df = pd.read_excel(CURRENT_DIR / "編號對應標準問句_增加類別251.xlsx").dropna(subset=["知識編號", "標準問句"])
# label_df["知識編號"] = label_df["知識編號"].astype(int)
# label_map = dict(zip(label_df["知識編號"], label_df["標準問句"]))

# query_df = pd.read_excel(CURRENT_DIR / "user_query.xlsx")

# # === QuerySystem ===
# class QuerySystem:
#     def __init__(self, builder: KnowledgeGraphBuilder, neighbor_weight=0.1, gamma=1.0,
#                  keyword_threshold=5, weight_strategy="constant"):
#         self.builder = builder
#         self.neighbor_weight = neighbor_weight
#         self.gamma = gamma
#         self.keyword_threshold = keyword_threshold
#         self.weight_strategy = weight_strategy
#         self.AC = ahocorasick.Automaton()
#         for kw in self.builder.keyword_to_questions.keys():
#             self.AC.add_word(kw, kw)
#         self.AC.make_automaton()

#     def get_weight_for_position(self, position):
#         if self.weight_strategy == "linear":
#             return 1 + position
#         elif self.weight_strategy == "increment":
#             return 1 + position * 0.5
#         elif self.weight_strategy == "small_increment":
#             return 1 + position * 0.1
#         else:
#             return 1

#     def get_keywords_with_weights(self, query):
#         keyword_scores = defaultdict(float)
#         query = query.lower()
#         matched_keywords = [kw for _, kw in self.AC.iter(query)]
#         use_weighting = len(matched_keywords) > self.keyword_threshold
#         if use_weighting:
#             for i, kw in enumerate(matched_keywords):
#                 keyword_scores[kw] += self.get_weight_for_position(i)
#         else:
#             for kw in matched_keywords:
#                 keyword_scores[kw] += 1
#         for kw in list(keyword_scores.keys()):
#             if kw in self.builder.graph:
#                 for neighbor in self.builder.graph.neighbors(kw):
#                     keyword_scores[neighbor] += self.neighbor_weight
#         return keyword_scores, use_weighting

#     def match_query_to_standard_question(self, query, normalize=True):
#         kw_with_scores, use_weighting = self.get_keywords_with_weights(query)
#         question_scores = defaultdict(float)
#         for kw, score in kw_with_scores.items():
#             qids = self.builder.keyword_to_questions.get(kw, [])
#             for qid in qids:
#                 question_scores[qid] += score
#         if normalize:
#             for qid in question_scores:
#                 num_keywords = len(self.builder.standard_questions.get(qid, []))
#                 if num_keywords > 0:
#                     question_scores[qid] /= (num_keywords ** self.gamma)
#         return sorted(question_scores.items(), key=lambda x: -x[1]), use_weighting

# # === 實驗流程 (只跑一組參數，並計算 Top-K / MRR / NDCG) ===
# def run_experiment_single():
#     builder = KnowledgeGraphBuilder(mu=MU)
#     builder.load_data_from_excel(CURRENT_DIR / "標準問句關鍵字_增加類別251.xlsx")
#     builder.build_graph()
#     system = QuerySystem(builder,
#                          neighbor_weight=NEIGHBOR_WEIGHT,
#                          gamma=GAMMA,
#                          keyword_threshold=KEYWORD_THRESHOLD,
#                          weight_strategy=WEIGHT_STRATEGY)

#     all_label_ids = list(label_map.keys())
#     all_label_texts = [label_map[qid] for qid in all_label_ids]

#     # 預先算候選向量
#     candidate_embs_dict = {}
#     for i in range(0, len(all_label_texts), 64):
#         batch_texts = all_label_texts[i:i+64]
#         batch_ids = all_label_ids[i:i+64]
#         batch_inputs = tokenizer_cl(batch_texts, return_tensors="pt", padding=True, truncation=True)
#         batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
#         with torch.no_grad():
#             batch_embs = cl_model(**batch_inputs)
#             if isinstance(batch_embs, (tuple, list)):
#                 batch_embs = batch_embs[0]
#             if batch_embs.dim() == 3:
#                 batch_embs = batch_embs[:, 0, :]
#             batch_embs = batch_embs.float()
#         for qid, emb in zip(batch_ids, batch_embs):
#             candidate_embs_dict[qid] = emb

#     # 記錄 Top-K / MRR / NDCG
#     topk_hits = {1:0, 3:0, 5:0}
#     mrr_list = []
#     ndcg1_list = []
#     ndcg3_list = []
#     ndcg5_list = []

#     def ndcg_from_rank(rank, k):
#         if rank is None:
#             return 0.0
#         return 1.0 / np.log2(rank + 1) if rank <= k else 0.0

#     for idx, row in tqdm(query_df.iterrows(), total=len(query_df), desc="推論", ncols=100):
#         query_sentence = str(row.get("text", "")).strip()
#         true_label = row.get("true_label", None)

#         # AC + Graph
#         matched_questions, _ = system.match_query_to_standard_question(query_sentence)
#         ac_top_qids = [qid for qid, _ in matched_questions[:AC_TOPN]]
#         if not ac_top_qids:
#             rank_index = None
#         else:
#             # CL 相似度
#             query_inputs = tokenizer_cl([query_sentence], return_tensors="pt", padding=True, truncation=True)
#             query_inputs = {k: v.to(device) for k, v in query_inputs.items()}
#             with torch.no_grad():
#                 query_emb = cl_model(**query_inputs)
#                 if isinstance(query_emb, (tuple, list)):
#                     query_emb = query_emb[0]
#                 if query_emb.dim() == 3:
#                     query_emb = query_emb[:, 0, :]
#                 query_emb = query_emb.float()

#             candidate_embs = torch.stack([candidate_embs_dict[qid] for qid in ac_top_qids], dim=0)
#             query_emb_norm = torch.nn.functional.normalize(query_emb, p=2, dim=1)
#             candidate_embs_norm = torch.nn.functional.normalize(candidate_embs, p=2, dim=1)
#             sims = torch.matmul(query_emb_norm, candidate_embs_norm.T)[0]
#             sims_cpu = sims.cpu().numpy()

#             qid_to_sim = {qid: sim for qid, sim in zip(ac_top_qids, sims_cpu)}
#             pred_label = max(qid_to_sim, key=qid_to_sim.get)
#             # Top-K 排序
#             ranked_labels = sorted(qid_to_sim.keys(), key=lambda x: -qid_to_sim[x])
#             try:
#                 rank_index = ranked_labels.index(true_label)
#                 rank = rank_index + 1
#             except ValueError:
#                 rank_index = None
#                 rank = None

#         # 計算 Top-K 命中
#         for k in topk_hits.keys():
#             if rank_index is not None and rank_index < k:
#                 topk_hits[k] += 1

#         # MRR
#         if rank_index is not None:
#             mrr_list.append(1.0 / (rank_index + 1))
#         else:
#             mrr_list.append(0.0)

#         # NDCG
#         ndcg1_list.append(ndcg_from_rank(rank_index+1 if rank_index is not None else None, 1))
#         ndcg3_list.append(ndcg_from_rank(rank_index+1 if rank_index is not None else None, 3))
#         ndcg5_list.append(ndcg_from_rank(rank_index+1 if rank_index is not None else None, 5))

#     total = len(query_df)
#     print("\n=== 指標結果 ===")
#     for k in [1,3,5]:
#         print(f"Top-{k}: {topk_hits[k]/total:.4f}")
#     print(f"MRR: {np.mean(mrr_list):.4f}")
#     print(f"NDCG@1: {np.mean(ndcg1_list):.4f}")
#     print(f"NDCG@3: {np.mean(ndcg3_list):.4f}")
#     print(f"NDCG@5: {np.mean(ndcg5_list):.4f}")

# # === 主程式 ===
# if __name__ == "__main__":
#     run_experiment_single()


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
import ast
from transformers import AutoConfig

# === 新增 NLTK 相關引用 (與 many_query_run_excel 一致) ===
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

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
    from model import CLBERT
except ImportError as e:
    print(f"匯入錯誤: {e}")
    print("請確認 build_graph.py 和 model 資料夾路徑是否正確")
    sys.exit(1)

warnings.filterwarnings('ignore')
# logging.set_verbosity_error()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === 檔案設定 ===
KG_SOURCE_FILE = CURRENT_DIR / "第九版關鍵字.xlsx" 
TEST_DATA_FILE = CURRENT_DIR / "user_query.xlsx"

# === 參數設定 (建議依照你 Grid Search 最佳結果調整) ===

# 50	0.1	0.7	0.7	0.9	0.4	3	small_increment

AC_TOPN = 50        # 依據 many_excel 的列表
NEIGHBOR_WEIGHT = 0.1
GAMMA=0.7
# RERANK_THRESHOLD = 0.7 # 關鍵閾值
RERANK_THRESHOLD = 0.7 # 關鍵閾值
ALPHA = 0.9         # 混合比例
MU = 0.4
KEYWORD_THRESHOLD = 3
WEIGHT_STRATEGY = "small_increment"


CONFIG = "hfl/chinese-roberta-wwm-ext"
DROPOUT = 0.3
MODEL_PATH = CURRENT_DIR.parent / "model" / "CL_student_2026-04-27-lamda09"

# === 1. 初始化模型 ===
print("Loading Model & Tokenizer...")
init_params = {'config': CONFIG, 'dropout': DROPOUT, 'num_class': None}
tokenizer_cl = AutoTokenizer.from_pretrained(init_params['config'], local_files_only=True)

if not os.path.exists(MODEL_PATH):
    print(f"警告: 找不到模型路徑 {MODEL_PATH}")
else:
    cl_model = CLBERT.from_pretrained(str(MODEL_PATH), args=init_params, local_files_only=True).to(device)
    cl_model.eval()

# === 2. 載入資料 ===
print(f"Loading Label Map from: {KG_SOURCE_FILE.name}")
label_df = pd.read_excel(KG_SOURCE_FILE)

# 自動偵測 ID 和 Text 欄位
possible_id_cols = ["id", "qid", "question_id", "編號", "label", "category_id"]
possible_text_cols = ["question", "text", "標準問句", "center_sentence", "sentence", "content", "central_text"]

found_id_col = next((col for col in label_df.columns if str(col).lower() in possible_id_cols), None)
found_text_col = next((col for col in label_df.columns if str(col).lower() in possible_text_cols), None)

if not found_id_col: found_id_col = label_df.columns[0]
if not found_text_col: found_text_col = label_df.columns[1]

print(f"Mapping Columns -> ID: '{found_id_col}', Text: '{found_text_col}'")

# 建立 Label Map
label_df = label_df.dropna(subset=[found_id_col, found_text_col])
label_df["_id_str"] = label_df[found_id_col].astype(str).str.replace(r'\.0$', '', regex=True)
label_map = dict(zip(label_df["_id_str"], label_df[found_text_col]))

# 讀取測試資料
print(f"Loading Test Data from: {TEST_DATA_FILE.name}")
query_df = pd.read_excel(TEST_DATA_FILE)

possible_gt_cols = ["true_label", "label", "target", "ground_truth", "qid"]
found_gt_col = next((col for col in query_df.columns if str(col).lower() in possible_gt_cols), None)

if found_gt_col:
    query_df["true_label"] = query_df[found_gt_col].astype(str).str.replace(r'\.0$', '', regex=True)
else:
    print("警告: 測試資料無 Label 欄位。")
    query_df["true_label"] = None

# === QuerySystem (完全移植自 many_query_run_excel.py) ===
class QuerySystem:
    def __init__(self, builder: KnowledgeGraphBuilder, neighbor_weight=0.1, gamma=1.0,
                 keyword_threshold=5, weight_strategy="constant"):
        self.builder = builder
        self.neighbor_weight = neighbor_weight
        self.gamma = gamma
        self.keyword_threshold = keyword_threshold
        self.weight_strategy = weight_strategy
        self.lemmatizer = WordNetLemmatizer()
        
        # --- 建立 AC 自動機 (支援一對多映射) ---
        self.AC = ahocorasick.Automaton()
        
        # 1. 建立 "處理後單字" -> "原始關鍵字集合" 的 mapping
        processed_to_original = defaultdict(set)
        for kw in self.builder.keyword_to_questions.keys():
            processed_kw = self.preprocess(kw)
            if processed_kw:
                processed_to_original[processed_kw].add(kw)
            
        # 2. 加入 AC
        for proc_kw, original_kws in processed_to_original.items():
            self.AC.add_word(proc_kw, original_kws)
            
        self.AC.make_automaton()

    def preprocess(self, text):
        if not isinstance(text, str):
            return ""
        return text.lower()

    def get_weight_for_position(self, position):
        if self.weight_strategy == "linear": return 1 + position
        elif self.weight_strategy == "increment": return 1 + position * 0.5
        elif self.weight_strategy == "small_increment": return 1 + position * 0.1
        else: return 1

    def get_keywords_with_weights(self, query):
        keyword_scores = defaultdict(float)
        
        # 1. 預處理 Query
        query_proc = self.preprocess(str(query))
        
        # 2. AC 匹配
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
        
        # 3. Graph Expansion
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

# === 實驗流程 ===
def run_experiment_single():
    print("Building Knowledge Graph...")
    builder = KnowledgeGraphBuilder(mu=MU)
    builder.load_data_from_excel(KG_SOURCE_FILE)
    builder.build_graph()

    try:
        print(f"[Graph] nodes = {builder.graph.number_of_nodes():,} | edges = {builder.graph.number_of_edges():,}")
    except Exception as e:
        print(f"[Graph] cannot read graph stats: {e}")

    system = QuerySystem(builder,
                         neighbor_weight=NEIGHBOR_WEIGHT,
                         gamma=GAMMA,
                         keyword_threshold=KEYWORD_THRESHOLD,
                         weight_strategy=WEIGHT_STRATEGY)

    # 預計算 Candidate Embeddings
    all_label_ids = list(label_map.keys())
    all_label_texts = [label_map[qid] for qid in all_label_ids]
    candidate_embs_dict = {}
    
    print("Encoding Candidates...")
    batch_size = 64
    for i in range(0, len(all_label_texts), batch_size):
        batch_texts = all_label_texts[i:i+batch_size]
        batch_ids = all_label_ids[i:i+batch_size]
        
        batch_inputs = tokenizer_cl(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
        
        with torch.no_grad():
            batch_embs = cl_model(**batch_inputs)
            if isinstance(batch_embs, (tuple, list)): batch_embs = batch_embs[0]
            if batch_embs.dim() == 3: batch_embs = batch_embs[:, 0, :]
            batch_embs = batch_embs.float()
            
        for qid, emb in zip(batch_ids, batch_embs):
            candidate_embs_dict[qid] = emb

    # === 指標變數初始化 ===
    topk_hits = {1:0, 3:0, 5:0}
    mrr_list = []
    ndcg1_list = []
    ndcg3_list = []
    ndcg5_list = []

    results = []

    def ndcg_from_rank(rank, k):
        if rank is None: return 0.0
        return 1.0 / np.log2(rank + 1) if rank <= k else 0.0

    print("Start Inference...")
    # =========================
    # Timing (mean ms/query)
    # =========================
    t_sum_no_topk_encode = 0.0     # A: 不含 topK 進 encoder
    t_sum_with_topk_encode = 0.0   # B: 含 topK 進 encoder（額外做）
    t_cnt = 0

    for idx, row in tqdm(query_df.iterrows(), total=len(query_df), ncols=100):
        query_text = row.get("text", row.get("query", row.get("question", "")))
        query_text = str(query_text).strip()
        true_label = row.get("true_label", None)

        if not true_label or true_label == "nan": continue
        if device.type == "cuda":
                    torch.cuda.synchronize()
        t0 = time.perf_counter()

        # 1. AC + Graph
        matched_questions, _ = system.match_query_to_standard_question(query_text)
        
        # 這裡根據 many_query 邏輯，如果沒匹配到，應該要保留空列表，但指標計算需要排名
        ac_top_qids = [qid for qid, _ in matched_questions[:AC_TOPN]]
        
        rank_index = None
        ranked_labels = []

        if ac_top_qids:
            # 2. CL Similarity
            q_inputs = tokenizer_cl([query_text], return_tensors="pt", padding=True, truncation=True, max_length=64)
            q_inputs = {k: v.to(device) for k, v in q_inputs.items()}
            
            with torch.no_grad():
                q_emb = cl_model(**q_inputs)
                if isinstance(q_emb, (tuple, list)): q_emb = q_emb[0]
                if q_emb.dim() == 3: q_emb = q_emb[:, 0, :]
                q_emb = q_emb.float()

            cand_embs = torch.stack([candidate_embs_dict[qid] for qid in ac_top_qids], dim=0)
                        # ---- Time A 截止點：不含 topK centers 進 encoder ----
            # （到這裡都還是用預先算好的 candidate_embs_dict）
            if device.type == "cuda":
                torch.cuda.synchronize()
            tA_end = time.perf_counter()

            # ---- Time B：額外把 topK centers 丟進 encoder（純比較用）----
            # 只做計時，不影響你原本的 pred / metrics
            topk_texts = [label_map[qid] for qid in ac_top_qids if qid in label_map]

            if len(topk_texts) > 0:
                topk_inputs = tokenizer_cl(
                    topk_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=64
                )
                topk_inputs = {k: v.to(device) for k, v in topk_inputs.items()}
                with torch.no_grad():
                    topk_out = cl_model(**topk_inputs)
                    if isinstance(topk_out, (tuple, list)):
                        topk_out = topk_out[0]
                    if topk_out.dim() == 3:
                        topk_out = topk_out[:, 0, :]
                    _ = topk_out.float()

            if device.type == "cuda":
                torch.cuda.synchronize()
            tB_end = time.perf_counter()
            
            q_emb_n = torch.nn.functional.normalize(q_emb, p=2, dim=1)
            c_embs_n = torch.nn.functional.normalize(cand_embs, p=2, dim=1)
            sims = torch.matmul(q_emb_n, c_embs_n.T)[0].cpu().numpy()

            qid_to_sim = {qid: sim for qid, sim in zip(ac_top_qids, sims)}
            
            # === 核心 Rerank 邏輯 (完全重現 many_query) ===
            # 先找 CL 分數最高的
            pred_label_cl = max(qid_to_sim, key=qid_to_sim.get)
            top_score_cl = qid_to_sim[pred_label_cl]
            
            final_scores = {}
            
            if top_score_cl < RERANK_THRESHOLD:
                # 啟動 Rerank: 混合 AC 分數
                sims_array = sims
                # CL 分數正規化 (Min-Max)
                sims_norm_vals = (sims_array - sims_array.min()) / (sims_array.max() - sims_array.min() + 1e-8)
                qid_to_sim_norm = {qid: val for qid, val in zip(ac_top_qids, sims_norm_vals)}

                # AC 分數正規化
                ac_score_dict = {qid: score for qid, score in matched_questions if qid in ac_top_qids}
                max_ac_score = max(ac_score_dict.values()) if ac_score_dict else 1.0
                ac_score_normalized = {qid: score / max_ac_score for qid, score in ac_score_dict.items()}

                # 計算混合分數
                for qid in ac_top_qids:
                    final_scores[qid] = ALPHA * qid_to_sim_norm[qid] + (1 - ALPHA) * ac_score_normalized.get(qid, 0)
            else:
                # 不啟動 Rerank，直接用 CL 分數
                final_scores = qid_to_sim
            
            # 根據最終分數排序
            ranked_labels = sorted(final_scores.keys(), key=lambda x: -final_scores[x])

            try:
                rank_index = ranked_labels.index(true_label)
                rank = rank_index + 1
            except ValueError:
                rank_index = None
                rank = None
            # 取得 Top-K 預測結果
            top1_label = ranked_labels[0] if len(ranked_labels) > 0 else None
            top1_text = label_map.get(top1_label, "") if top1_label is not None else ""
            top1_score = final_scores[top1_label] if top1_label is not None else None

            top3_labels = ranked_labels[:3]
            top3_texts = [label_map.get(qid, "") for qid in top3_labels]
            top3_scores = [final_scores[qid] for qid in top3_labels]

            top5_labels = ranked_labels[:5]
            top5_texts = [label_map.get(qid, "") for qid in top5_labels]
            top5_scores = [final_scores[qid] for qid in top5_labels]

            results.append({
                "text": query_text,
                "true_label": true_label,
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
        else:
            rank_index = None
            if device.type == "cuda":
                torch.cuda.synchronize()
            tA_end = time.perf_counter()
            tB_end = tA_end
            rank = None
            results.append({
                "text": query_text,
                "true_label": true_label,
                "pred_top1_label": None,
                "pred_top1_text": "",
                "pred_top1_score": None,
                "top3_labels": "[]",
                "top3_texts": "[]",
                "top3_scores": "[]",
                "top5_labels": "[]",
                "top5_texts": "[]",
                "top5_scores": "[]",
                "top1_correct": 0,
                "rank_of_true_label": None,
                "reciprocal_rank": 0.0
            })
        # ---- accumulate timing (only for valid queries) ----
        t_sum_no_topk_encode += (tA_end - t0)
        t_sum_with_topk_encode += (tB_end - t0)
        t_cnt += 1
        # === 計算指標 ===
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

    total = len(ndcg1_list)
    if total == 0:
        print("沒有有效的測試樣本。")
        return

    print("\n=== 指標結果 ===")
    print(f"Top-1: {topk_hits[1]/total:.4f}")
    print(f"Top-3: {topk_hits[3]/total:.4f}")
    print(f"Top-5: {topk_hits[5]/total:.4f}")
    print(f"MRR:   {np.mean(mrr_list):.4f}")
    print(f"NDCG@1: {np.mean(ndcg1_list):.4f}")
    print(f"NDCG@3: {np.mean(ndcg3_list):.4f}")
    print(f"NDCG@5: {np.mean(ndcg5_list):.4f}")
    # =========================
    # Timing report
    # =========================
    if t_cnt > 0:
        mean_ms_A = (t_sum_no_topk_encode / t_cnt) * 1000.0
        mean_ms_B = (t_sum_with_topk_encode / t_cnt) * 1000.0
        print(f"\n=== 推論時間 (mean ms/query, 不含KG建立) ===")
        print(f"A) 不含 topK centers 進 encoder: {mean_ms_A:.3f} ms/query  (N={t_cnt})")
        print(f"B) 含 topK centers 進 encoder:   {mean_ms_B:.3f} ms/query  (N={t_cnt})")
        results_df = pd.DataFrame(results)
        output_file = CURRENT_DIR / "prediction_results.xlsx"
        results_df.to_excel(output_file, index=False, engine="openpyxl")
        print(f"\n已輸出每筆預測結果：{output_file}")
if __name__ == "__main__":
    run_experiment_single()