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
AC_TOPN_LIST = [80]
NEIGHBOR_WEIGHT_LIST = [0.1]
RERANK_THRESHOLD_LIST = [0.1]
ALPHA_LIST = [0.1]
MU_LIST = [0.2]
KEYWORD_THRESHOLD_LIST = [3]
WEIGHT_STRATEGY_LIST = [ "increment"]
GAMMA_LIST = [1]

# 針對特定關鍵字再加分
SELECTED_KEYWORDS = ["申訴", "不合理", "不服", "有問題"]
SPECIAL_MAIN_WEIGHT_LIST = [0]
SPECIAL_NEIGHBOR_WEIGHT_LIST = [0]

# 針對特定標準問句強制納入候選 & 額外加分
SPECIAL_STANDQ_ID_LIST = [[]]
SPECIAL_STANDQ_WEIGHT_LIST = [0]

# === 【新增】類別加分參數 (Category Bonus) ===
# 當 query 命中某個類別關鍵字時，該類別下的所有標準問句額外加分
CATEGORY_BONUS_LIST = [1] 

# 額外選項
PRINT_GRAPH_STATS = True 

# === 固定參數 ===
CONFIG = "hfl/chinese-roberta-wwm-ext"
DROPOUT = 0.3
NUM_CLASS = None
MODEL_PATH = ROOT_DIR / "model" / "CL_2025-10-25"

# 輸出詳細結果的資料夾
datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
out_detal_file = "10-25_v13_detail_catbonus_" + datetime_str  # 修改檔名以便識別
OUT_DETAIL_DIR = CURRENT_DIR / out_detal_file
OUT_DETAIL_DIR.mkdir(exist_ok=True)

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
label_df = pd.read_excel(CURRENT_DIR / "編號對應標準問句_增加類別251.xlsx")
label_df = label_df.dropna(subset=["知識編號", "標準問句"])
label_df["知識編號"] = label_df["知識編號"].astype(int)
label_map = dict(zip(label_df["知識編號"], label_df["標準問句"]))

query_df = pd.read_excel(CURRENT_DIR / "1030_驗證題.xlsx")


# === QuerySystem ===
class QuerySystem:
    def __init__(self, builder: KnowledgeGraphBuilder, neighbor_weight=0.1, gamma=1.0,
                 keyword_threshold=5, weight_strategy="constant",
                 selected_keywords=None, special_main=0.0, special_neighbor=0.0,
                 category_keywords=None, category_bonus=0.0): # 新增 category 參數
        self.builder = builder
        self.neighbor_weight = neighbor_weight
        self.gamma = gamma
        self.keyword_threshold = keyword_threshold
        self.weight_strategy = weight_strategy
        self.selected_keywords = selected_keywords or []
        self.special_main = special_main
        self.special_neighbor = special_neighbor
        
        # === 【新增】類別加分設定 ===
        self.category_keywords = category_keywords or set()
        self.category_bonus = category_bonus

        self.AC = ahocorasick.Automaton()
        for kw in self.builder.keyword_to_questions.keys():
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
        
    def match_query_to_standard_question(self, query, normalize=True, gamma=1.0):
        kw_with_scores, use_weighting, special_keyword_hit = self.get_keywords_with_weights(query)
        question_scores = defaultdict(float)
        
        for kw, score in kw_with_scores.items():
            qids = self.builder.keyword_to_questions.get(kw, [])
            
            # === 【新增】檢查是否為類別關鍵字 ===
            is_category = kw in self.category_keywords
            
            for qid in qids:
                question_scores[qid] += score
                
                # 如果這個關鍵字是類別，且該問句屬於這個類別，額外加分
                if is_category:
                    question_scores[qid] += self.category_bonus

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


# === 實驗主程式 ===
def run_experiment(mu, neighbor_weight, ac_topn, rerank_threshold, alpha,
                   keyword_threshold, weight_strategy, special_main, special_neighbor, 
                   special_standq_ids, special_standq_weight, gamma,
                   category_bonus): # 新增接收 category_bonus
    
    builder = KnowledgeGraphBuilder(mu=mu)
    builder.load_data_from_excel(CURRENT_DIR / "第十版關鍵字.xlsx")
    builder.build_graph()
    
    # 用來記錄哪些 keyword 其實是 category
    detected_category_keywords = set()

    # =========== 【新增開始】注入類別資料 ===========
    possible_files = [
        CURRENT_DIR / "類別對應標準問句.xlsx"
    ]
    
    target_file = None
    for f in possible_files:
        if f.exists():
            target_file = f
            break
            
    if target_file:
        try:
            if target_file.suffix.lower().endswith('.csv') or 'csv' in target_file.name.lower():
                cat_df = pd.read_csv(target_file)
            else:
                cat_df = pd.read_excel(target_file)

            cat_df.columns = [str(c).strip() for c in cat_df.columns]
            
            if '主題' in cat_df.columns and '編號' in cat_df.columns:
                cat_df = cat_df.dropna(subset=['主題', '編號'])
                
                count_loaded = 0
                for _, row_cat in cat_df.iterrows():
                    category_keyword = str(row_cat['主題']).strip()
                    # === 收集類別關鍵字 ===
                    detected_category_keywords.add(category_keyword)

                    try:
                        qid = int(row_cat['編號'])
                    except ValueError:
                        continue 

                    # 1. 更新 keyword_to_questions 映射表
                    try:
                        if category_keyword not in builder.keyword_to_questions:
                            builder.keyword_to_questions[category_keyword] = set()
                        
                        container = builder.keyword_to_questions[category_keyword]
                        
                        if isinstance(container, set):
                            container.add(qid)
                        elif isinstance(container, list):
                            if qid not in container:
                                container.append(qid)
                        count_loaded += 1
                    except Exception as e:
                        pass 

                    # 2. 更新 Graph 圖結構
                    if hasattr(builder, 'graph'):
                        try:
                            if not builder.graph.has_node(category_keyword):
                                builder.graph.add_node(category_keyword, type='keyword')
                            
                            if builder.graph.has_node(qid):
                                builder.graph.add_edge(category_keyword, qid, weight=1.0)
                        except Exception:
                            pass
            else:
                print(f"警告: {target_file.name} 欄位不符，需包含 '主題' 與 '編號'")
                
        except Exception as e:
            print(f"讀取類別檔案時發生錯誤: {e}")
    else:
        print("提示: 未找到 '類別對應標準問句' 相關檔案，跳過類別注入。")
    # =========== 【新增結束】 ===========

    system = QuerySystem(builder,
                         neighbor_weight=neighbor_weight,
                         gamma=gamma,
                         keyword_threshold=keyword_threshold,
                         weight_strategy=weight_strategy,
                         selected_keywords=SELECTED_KEYWORDS,
                         special_main=special_main,
                         special_neighbor=special_neighbor,
                         category_keywords=detected_category_keywords, # 傳入類別集合
                         category_bonus=category_bonus) # 傳入加分權重


    results_all = [] 

    for idx, row in query_df.iterrows():
        query_sentence = str(row.get("text", "")).strip()
        true_label = row.get("true_label", None)
        start_time = time.time()

        # 1) AC + 圖擴散 + 類別加分
        matched_questions, use_weighting, special_keyword_hit = system.match_query_to_standard_question(query_sentence, normalize=True, gamma=gamma)
        ac_ranked = [qid for qid, _ in matched_questions]
        ac_top_qids = ac_ranked[:ac_topn] 

        # === 強制加入特定標準問句 ===
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
        
        if not ac_top_qids:
            continue 

        # 2) 語意模型
        with torch.no_grad():
            query_inputs = tokenizer_cl([query_sentence], return_tensors="pt", padding=True, truncation=True).to(device)
            query_emb = cl_model(**query_inputs)
            if isinstance(query_emb, (tuple, list)):
                query_emb = query_emb[0]
            if query_emb.dim() == 3:
                query_emb = query_emb[:, 0, :]
            
            candidate_texts = [label_map[qid] for qid in ac_top_qids]
            candidate_inputs = tokenizer_cl(candidate_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            candidate_embs = cl_model(**candidate_inputs)
            if isinstance(candidate_embs, (tuple, list)):
                candidate_embs = candidate_embs[0]
            if candidate_embs.dim() == 3:
                candidate_embs = candidate_embs[:, 0, :]
        
        sims = cosine_similarity(query_emb.cpu().numpy(), candidate_embs.cpu().numpy())[0]
        qid_to_sim = {qid: sim for qid, sim in zip(ac_top_qids, sims)}
        pred_label_cl = max(qid_to_sim, key=qid_to_sim.get)
        top_score_cl = qid_to_sim[pred_label_cl]

        pred_label_final = pred_label_cl
        top_score_final = top_score_cl
        rerank_flag = 0

        # 3) Rerank
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

        results_all.append({
            "query_id": idx + 1,
            "query_text": query_sentence,
            "true_standard_question": label_map.get(true_label, "未知"),
            "predicted_standard_question": label_map.get(pred_label_final, "未知"),
            "是否正確": int(pred_label_final == true_label),
            "是否用rerank": rerank_flag,
            "CL過濾TOP1": label_map.get(pred_label_cl, "未知"),
            "CL過濾TOP1分數": top_score_cl,
            "最終預測結果": label_map.get(pred_label_final, "未知"),
            "最終預測分數": top_score_final,
            "耗時(秒)": time.time() - start_time
        })

    out_df = pd.DataFrame(results_all)
    evaluated = len(out_df)
    correct_count = out_df["是否正確"].sum() if evaluated > 0 else 0
    accuracy = (correct_count / evaluated) if evaluated > 0 else 0
    coverage = (evaluated / len(query_df)) if len(query_df) > 0 else 0
    if correct_count > 67:
        print(correct_count)
    
    # 檔名增加 cat_bonus 記錄
    # 更新後的檔名，包含所有參數
    detail_name = (
        f"res_AC{ac_topn}_"
        f"catB{category_bonus}_"
        f"th{rerank_threshold}_"
        f"alpha{alpha}_"
        f"mu{mu}_"
        f"kt{keyword_threshold}_"
        f"ws{weight_strategy}_"
        f"sm{special_main}_"
        f"sn{special_neighbor}_"
        f"sid{special_standq_ids}_"
        f"sw{special_standq_weight}_"
        f"gam{gamma}.xlsx"
    )
    out_df.to_excel(OUT_DETAIL_DIR / detail_name, index=False)

    return correct_count, accuracy, coverage, evaluated


if __name__ == "__main__":
    summary_records = []

    # 參數組合，加入 category_bonus
    param_combinations = [
    (ac_topn, neighbor_weight, rerank_threshold, alpha, mu,
     keyword_threshold, weight_strategy, special_main,
     special_neighbor, special_standq_ids, special_standq_weight, gamma, category_bonus)
    for ac_topn, neighbor_weight, rerank_threshold, alpha, mu,
        keyword_threshold, weight_strategy, special_main,
        special_neighbor, special_standq_ids, special_standq_weight, gamma, category_bonus
    in product(
        AC_TOPN_LIST,
        NEIGHBOR_WEIGHT_LIST,
        RERANK_THRESHOLD_LIST,
        ALPHA_LIST,
        MU_LIST,
        KEYWORD_THRESHOLD_LIST,
        WEIGHT_STRATEGY_LIST,
        SPECIAL_MAIN_WEIGHT_LIST,
        SPECIAL_NEIGHBOR_WEIGHT_LIST,
        SPECIAL_STANDQ_ID_LIST,
        SPECIAL_STANDQ_WEIGHT_LIST,
        GAMMA_LIST,
        CATEGORY_BONUS_LIST # 加入這裡
    )
    if special_main >= special_neighbor
    ]

    for ac_topn, neighbor_weight, rerank_threshold, alpha, mu, keyword_threshold, weight_strategy, special_main, special_neighbor, special_standq_ids, special_standq_weight, gamma, category_bonus in tqdm(
            param_combinations, desc="Running experiments", dynamic_ncols=True, leave=True, ncols=100):
        
        correct, acc, cov, evaluated = run_experiment(
            mu, neighbor_weight, ac_topn, rerank_threshold, alpha,
            keyword_threshold, weight_strategy, special_main, special_neighbor,
            special_standq_ids, special_standq_weight, gamma,
            category_bonus # 傳入參數
        )

        summary_records.append({
            "ac_filter_size": ac_topn,
            "category_bonus": category_bonus, # 記錄於報表
            "neighbor_weight": neighbor_weight,
            "rerank_threshold": rerank_threshold,
            "rerank_alpha": alpha,
            "mu": mu,
            "keyword_threshold": keyword_threshold,
            "weight_strategy": weight_strategy,
            "special_main": special_main, 
            "special_neighbor": special_neighbor,
            "special_standq_ids": special_standq_ids,
            "special_standq_weight": special_standq_weight,
            "gamma": gamma,
            "category_bonus": category_bonus,
            "Correct Count": correct,
            "Evaluated": evaluated,
            "Coverage (%)": round(cov * 100, 2),
            "Answer_Accuracy (%)": round(acc * 100, 2)
        })

    ac_str = "_".join(map(str, AC_TOPN_LIST))
    outfile = f"10-25_v13_summary_{datetime_str}_AC{ac_str}.xlsx"
    pd.DataFrame(summary_records).to_excel(CURRENT_DIR / outfile, index=False)
    print(f"完成，輸出檔案: {outfile}")