import os
import sys
import time
import torch
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import product
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, logging
import ahocorasick
from tqdm import tqdm
from datetime import datetime

# === 路徑設定 ===
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
sys.path.append(str(ROOT_DIR))
sys.path.append(str(ROOT_DIR / "model"))

from build_graph import KnowledgeGraphBuilder
from model import CLBERT

# === Grid Search 參數空間 ===
# 建議將 MU_LIST 放在最外層迴圈（或排序時以此為主），減少建圖次數
AC_TOPN_LIST = [1, 5, 10, 20, 40, 60, 80, 100 ]
CL_TOPN_LIST = [1, 5, 10, 20, 40, 60, 80, 100 ]
NEIGHBOR_WEIGHT_LIST = [0.1]
ALPHA_LIST = [0.1, 0.3, 0.5, 0.7, 0.9]
MU_LIST = [0.2, 0.4, 0.8]
KEYWORD_THRESHOLD_LIST = [3, 5, 7]
WEIGHT_STRATEGY_LIST = ["small_increment", "increment", "linear", "constant"]
GAMMA_LIST = [0, 0.1, 0.3, 0.7, 0.9,1]

# 針對特定關鍵字再加分
SELECTED_KEYWORDS = ["申訴", "不合理", "不服", "有問題"]
SPECIAL_MAIN_WEIGHT_LIST = [0]
SPECIAL_NEIGHBOR_WEIGHT_LIST = [0]
SPECIAL_STANDQ_ID_LIST = [[]]
SPECIAL_STANDQ_WEIGHT_LIST = [0]

# 額外選項
PRINT_GRAPH_STATS = False # 跑大量時建議關閉

# === 固定參數 ===
CONFIG = "hfl/chinese-roberta-wwm-ext"
DROPOUT = 0.3
NUM_CLASS = None
MODEL_PATH = ROOT_DIR / "model" / "CL_2025-10-03"
BATCH_SIZE = 128  # 增大 Batch size 加速預計算

datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
out_detal_file = "detail_acfilter_" + datetime_str
OUT_DETAIL_DIR = CURRENT_DIR / out_detal_file
OUT_DETAIL_DIR.mkdir(exist_ok=True)

logging.set_verbosity_error()
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 載入模型 & 資料 (全域執行一次) ===
print("初始化模型與載入資料中...")
init_params = {'config': CONFIG, 'dropout': DROPOUT, 'num_class': NUM_CLASS}
tokenizer_cl = AutoTokenizer.from_pretrained(init_params['config'], local_files_only=True)
cl_model = CLBERT.from_pretrained(str(MODEL_PATH), args=init_params, local_files_only=True).to(device)
cl_model.eval()

# 載入 Excel
label_df = pd.read_excel(CURRENT_DIR / "編號對應標準問句_增加類別251.xlsx")
label_df = label_df.dropna(subset=["知識編號", "標準問句"])
label_df["知識編號"] = label_df["知識編號"].astype(int)
label_map = dict(zip(label_df["知識編號"], label_df["標準問句"]))

# 確保順序固定
ALL_QIDS = list(label_map.keys())
ALL_LABEL_TEXTS = [label_map[qid] for qid in ALL_QIDS]

query_df = pd.read_excel(CURRENT_DIR / "1030_驗證題.xlsx")
ALL_QUERY_TEXTS = [str(t).strip() for t in query_df["text"].tolist()]

# === 核心優化：預先計算所有 Embedding 與 相似度矩陣 ===
def get_embeddings(texts, batch_size=BATCH_SIZE):
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer_cl(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            embs = cl_model(**inputs)
            if isinstance(embs, (tuple, list)):
                embs = embs[0]
            if embs.dim() == 3:
                embs = embs[:, 0, :]
            all_embs.append(embs.cpu().numpy())
    return np.vstack(all_embs)

print("正在預計算 Embedding 與 相似度矩陣 (這只需要做一次)...")
# 1. 計算所有標準問句 Embedding (N_labels, 768)
label_embs = get_embeddings(ALL_LABEL_TEXTS)
# 2. 計算所有測試問句 Embedding (N_queries, 768)
query_embs = get_embeddings(ALL_QUERY_TEXTS)

# 3. 計算完整相似度矩陣 (N_queries, N_labels)
# SIM_MATRIX[i][j] 代表 第 i 個 query 對 第 j 個 standard question 的相似度
SIM_MATRIX = cosine_similarity(query_embs, label_embs)
print("預計算完成！開始 Grid Search...")

# === QuerySystem ===
class QuerySystem:
    def __init__(self, builder, neighbor_weight=0.1, gamma=1.0,
                 keyword_threshold=5, weight_strategy="constant",
                 selected_keywords=None, special_main=0.0, special_neighbor=0.0):
        self.builder = builder
        self.neighbor_weight = neighbor_weight
        self.gamma = gamma
        self.keyword_threshold = keyword_threshold
        self.weight_strategy = weight_strategy
        self.selected_keywords = selected_keywords or []
        self.special_main = special_main
        self.special_neighbor = special_neighbor

        self.AC = ahocorasick.Automaton()
        for kw in self.builder.keyword_to_questions.keys():
            self.AC.add_word(kw, kw)
        self.AC.make_automaton()

    def get_weight_for_position(self, position):
        if self.weight_strategy == "linear": return 1 + position
        elif self.weight_strategy == "increment": return 1 + position * 0.5
        elif self.weight_strategy == "small_increment": return 1 + position * 0.1
        else: return 1

    def match_query_to_standard_question(self, query, normalize=True, gamma=1.0):
        kw_with_scores, use_weighting, special_keyword_hit = self.get_keywords_with_weights(query)
        question_scores = defaultdict(float)
        for kw, score in kw_with_scores.items():
            qids = self.builder.keyword_to_questions.get(kw, [])
            for qid in qids:
                question_scores[qid] += score
        if normalize:
            for qid in question_scores:
                num_keywords = len(self.builder.standard_questions.get(qid, []))
                if num_keywords > 0:
                    question_scores[qid] /= (num_keywords ** gamma)
        return sorted(question_scores.items(), key=lambda x: -x[1]), use_weighting, special_keyword_hit

    def get_keywords_with_weights(self, query, filter_keywords=False):
        keyword_scores = defaultdict(float)
        matched_keywords = []
        special_keyword_hit = False
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
            if kw in self.builder.graph:
                for neighbor in self.builder.graph.neighbors(kw):
                    keyword_scores[neighbor] += self.neighbor_weight

        for kw in self.selected_keywords:
            if kw in matched_keywords:
                keyword_scores[kw] += self.special_main
                special_keyword_hit = True
            else:
                for matched_kw in matched_keywords:
                    if matched_kw in self.builder.graph and kw in self.builder.graph.neighbors(matched_kw):
                        keyword_scores[kw] += self.special_neighbor
                        special_keyword_hit = True
                        break
        
        if filter_keywords:
            keyword_scores = {kw: score for kw, score in keyword_scores.items() if kw in self.selected_keywords}
        return keyword_scores, use_weighting, special_keyword_hit

# === run_experiment 優化版 (接收預計算的矩陣與現成 System) ===
def run_experiment(system, sim_matrix, 
                   ac_topn, alpha, special_standq_ids, special_standq_weight, gamma, cl_topn_full):
    
    results_all = []
    
    # 這裡直接用 zip 跑，速度最快
    for idx, (query_text, true_label) in enumerate(zip(ALL_QUERY_TEXTS, query_df["true_label"])):
        start_time = time.time()
        
        # 1. 取得預計算的相似度 (O(1) 查表)
        sims_all = sim_matrix[idx] #這行取代了原本幾秒鐘的模型運算

        # 1c. 取 CL Top-N
        # argpartition 比 argsort 快，因為只需要前 N 個
        if cl_topn_full < len(sims_all):
            cl_top_idx = np.argpartition(-sims_all, cl_topn_full)[:cl_topn_full]
        else:
            cl_top_idx = np.arange(len(sims_all))
        cl_top_qids = [ALL_QIDS[i] for i in cl_top_idx]

        # 2. AC 檢索
        matched_questions, use_weighting, special_keyword_hit = system.match_query_to_standard_question(query_text, normalize=True, gamma=gamma)
        ac_ranked = [qid for qid, _ in matched_questions]
        ac_top_qids = ac_ranked[:ac_topn]

        # Special logic
        if special_standq_ids and special_keyword_hit:
            for sid in special_standq_ids:
                if sid not in ac_top_qids and sid in label_map:
                    ac_top_qids.append(sid)
                    if sid not in dict(matched_questions):
                        matched_questions.append((sid, 0.0))
                    for i, (qid, score) in enumerate(matched_questions):
                        if qid == sid:
                            matched_questions[i] = (qid, score + special_standq_weight)
                            break

        # 3. Union
        candidate_qids = list(dict.fromkeys(cl_top_qids + ac_top_qids))
        if not candidate_qids:
            continue

        # 4. 提取分數
        qid_to_sim = {qid: float(sims_all[ALL_QIDS.index(qid)]) for qid in candidate_qids}
        ac_score_map = {qid: score for qid, score in matched_questions}
        qid_to_kw = {qid: float(ac_score_map.get(qid, 0.0)) for qid in candidate_qids}

        # 5. Normalize
        sim_vals = np.array(list(qid_to_sim.values()), dtype=float)
        sim_min, sim_max = sim_vals.min(), sim_vals.max()
        sim_norm_map = {qid: 0.5 if sim_max == sim_min else (qid_to_sim[qid] - sim_min) / (sim_max - sim_min) for qid in candidate_qids}

        kw_vals = np.array(list(qid_to_kw.values()), dtype=float)
        kw_min, kw_max = kw_vals.min(), kw_vals.max()
        kw_norm_map = {qid: 0.5 if kw_max == kw_min else (qid_to_kw[qid] - kw_min) / (kw_max - kw_min) for qid in candidate_qids}

        # 6. Hybrid Score
        score_map = {}
        for qid in candidate_qids:
            score_map[qid] = alpha * sim_norm_map[qid] + (1 - alpha) * kw_norm_map[qid]

        top1_qid, top1_score = sorted(score_map.items(), key=lambda x: -x[1])[0]

        results_all.append({
            "query_id": idx + 1,
            "query_text": query_text,
            "true_standard_question": label_map.get(true_label, "未知"),
            "candidate_count": len(candidate_qids),
            "top1_qid": top1_qid,
            "top1_text": label_map.get(top1_qid, "未知"),
            "top1_score": top1_score,
            "是否正確": int(top1_qid == true_label) if pd.notna(true_label) else 0,
            "耗時(秒)": time.time() - start_time
        })

    # 匯總
    out_df = pd.DataFrame(results_all)
    evaluated = len(out_df)
    correct_count = out_df["是否正確"].sum() if evaluated > 0 else 0
    accuracy = (correct_count / evaluated) if evaluated > 0 else 0
    coverage = (evaluated / len(query_df)) if len(query_df) > 0 else 0

    if correct_count>63:
        print(correct_count)
    # 寫入檔案
    detail_name = f"AC{ac_topn}_CL{cl_topn_full}_neighbor{system.neighbor_weight}_alpha{alpha}_mu{system.builder.mu}_kt{system.keyword_threshold}_{system.weight_strategy}_sm{system.special_main}_sn{system.special_neighbor}_sid{','.join(map(str, special_standq_ids or []))}_w{special_standq_weight}_gamma{gamma}.xlsx"
    out_df.to_excel(OUT_DETAIL_DIR / detail_name, index=False)

    return correct_count, accuracy, coverage, evaluated

if __name__ == "__main__":
    summary_records = []

    # 產生參數組合
    all_params = list(product(
        AC_TOPN_LIST, CL_TOPN_LIST, NEIGHBOR_WEIGHT_LIST, ALPHA_LIST, MU_LIST,
        KEYWORD_THRESHOLD_LIST, WEIGHT_STRATEGY_LIST, SPECIAL_MAIN_WEIGHT_LIST,
        SPECIAL_NEIGHBOR_WEIGHT_LIST, SPECIAL_STANDQ_ID_LIST, SPECIAL_STANDQ_WEIGHT_LIST, GAMMA_LIST
    ))
    all_params = [p for p in all_params if p[7] >= p[8]] # special_main >= special_neighbor check

    # === 關鍵優化：依照 mu 排序參數 ===
    # 這樣可以讓 graph builder 盡量少重建
    # mu 在 index 4, neighbor_weight 在 index 2, keyword_threshold 在 index 5
    # 我們把會影響 Graph 結構的參數當作主要排序鍵
    all_params.sort(key=lambda x: (x[4], x[2], x[5], x[6], x[7], x[8])) 

    current_builder_params = None
    current_system = None
    
    # 為了快取 Builder (避免重複讀取 Excel)
    cached_builder = None 

    pbar = tqdm(all_params, desc="Experiments", dynamic_ncols=True, ncols=120)
    
    for params in pbar:
        (ac_topn, cl_topn, neighbor_weight, alpha, mu,
         keyword_threshold, weight_strategy, special_main,
         special_neighbor, special_standq_ids, special_standq_weight, gamma) = params
        
        # 1. 檢查是否需要重建 Graph / System
        # 影響 Graph 的參數: mu
        # 影響 System (AC) 的參數: neighbor_weight, keyword_threshold, weight_strategy, keywords
        
        # 簡單起見，如果 mu 變了，重建 Builder
        if cached_builder is None or cached_builder.mu != mu:
            cached_builder = KnowledgeGraphBuilder(mu=mu)
            cached_builder.load_data_from_excel(CURRENT_DIR / "第九版_關鍵字(暫定).xlsx")
            cached_builder.build_graph()
        
        # 如果 System 參數變了，重建 System
        # 這裡簡單判定：只要 builder 變了或者 system 參數變了就重建
        # 其實 System 建置很快，可以每次都建，但為了極致優化：
        system_params_key = (mu, neighbor_weight, keyword_threshold, weight_strategy, special_main, special_neighbor, gamma)
        
        # 注意：我們直接每次重建 System，因為它依賴 gamma 和 builder，而且建立 System 遠比跑模型快
        # 但為了不讓 AC 每次重建，我們可以稍微檢查一下
        if current_system is None or \
           current_system.builder.mu != mu or \
           current_system.neighbor_weight != neighbor_weight or \
           current_system.keyword_threshold != keyword_threshold or \
           current_system.weight_strategy != weight_strategy or \
           current_system.special_main != special_main or \
           current_system.special_neighbor != special_neighbor:
           
            current_system = QuerySystem(cached_builder,
                                         neighbor_weight=neighbor_weight,
                                         gamma=gamma, # Update gamma dynamically inside function if needed, but QuerySystem keeps it
                                         keyword_threshold=keyword_threshold,
                                         weight_strategy=weight_strategy,
                                         selected_keywords=SELECTED_KEYWORDS,
                                         special_main=special_main,
                                         special_neighbor=special_neighbor)
        
        # 2. 執行實驗 (傳入預計算矩陣)
        correct, acc, cov, evaluated = run_experiment(
            current_system, SIM_MATRIX,
            ac_topn, alpha, special_standq_ids, special_standq_weight, gamma,
            cl_topn_full=cl_topn
        )

        summary_records.append({
            "ac_filter_size": ac_topn,
            "cl_topn_full": cl_topn,
            "neighbor_weight": neighbor_weight,
            "rerank_alpha": alpha,
            "mu": mu,
            "keyword_threshold": keyword_threshold,
            "weight_strategy": weight_strategy,
            "special_main": special_main,
            "special_neighbor": special_neighbor,
            "special_standq_ids": special_standq_ids,
            "special_standq_weight": special_standq_weight,
            "gamma": gamma,
            "Correct Count": correct,
            "Evaluated": evaluated,
            "Coverage (%)": round(cov * 100, 2),
            "Answer_Accuracy (%)": round(acc * 100, 2)
        })

    ac_str = "_".join(map(str, AC_TOPN_LIST))
    outfile = f"summary_{datetime_str}_AC{ac_str}.xlsx"
    pd.DataFrame(summary_records).to_excel(CURRENT_DIR / outfile, index=False)
    print(f"完成，輸出檔案: {outfile}")