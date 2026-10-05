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
AC_TOPN_LIST = [80, 100]
NEIGHBOR_WEIGHT_LIST = [0.1]
RERANK_THRESHOLD_LIST = [0.1, 0.3, 0.5, 0.7, 0.9, 1]
ALPHA_LIST = [0.1, 0.3, 0.5] # 還沒跑0.7, 0.9, 1
MU_LIST = [0.2, 0.4, 0.8]
KEYWORD_THRESHOLD_LIST = [3, 5, 7]
WEIGHT_STRATEGY_LIST = ["small_increment","increment", "linear"]

SELECTED_KEYWORDS = ["申訴", "不合理", "不服", "有問題"]
SPECIAL_MAIN_WEIGHT_LIST = [0.5, 1, 2]
SPECIAL_NEIGHBOR_WEIGHT_LIST = [0.1, 0.5, 1]

SPECIAL_STANDQ_ID_LIST = [[2], [1, 2]]
SPECIAL_STANDQ_WEIGHT_LIST = [1,2,3]

PRINT_GRAPH_STATS = True
CONFIG = "hfl/chinese-roberta-wwm-ext"
DROPOUT = 0.3
NUM_CLASS = None
MODEL_PATH = ROOT_DIR / "model" / "CL_2025-10-25"

# === 設定是否接續舊實驗資料夾 ===
resume_folder = "detail_acfilter_20251119_134426"
if resume_folder and (CURRENT_DIR / resume_folder).exists():
    OUT_DETAIL_DIR = CURRENT_DIR / resume_folder
    print(f"🔁 偵測到舊資料夾，將繼續執行：{resume_folder}")
else:
    datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_detal_file = "detail_acfilter_" + datetime_str
    OUT_DETAIL_DIR = CURRENT_DIR / out_detal_file
    OUT_DETAIL_DIR.mkdir(exist_ok=True)
    print(f"🆕 建立新資料夾：{OUT_DETAIL_DIR.name}")
    resume_folder = OUT_DETAIL_DIR.name

datetime_str = resume_folder.split("_")[-1] if "_" in resume_folder else datetime.now().strftime('%Y%m%d_%H%M%S')

logging.set_verbosity_error()
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 初始化模型 ===
init_params = {'config': CONFIG, 'dropout': DROPOUT, 'num_class': NUM_CLASS}
tokenizer_cl = AutoTokenizer.from_pretrained(init_params['config'], local_files_only=True)
cl_model = CLBERT.from_pretrained(str(MODEL_PATH), args=init_params, local_files_only=True).to(device)
cl_model.eval()

# === 載入資料 ===
label_df = pd.read_excel(CURRENT_DIR / "編號對應標準問句_增加類別251.xlsx")
label_df = label_df.dropna(subset=["知識編號", "標準問句"])
label_df["知識編號"] = label_df["知識編號"].astype(int)
label_map = dict(zip(label_df["知識編號"], label_df["標準問句"]))
query_df = pd.read_excel(CURRENT_DIR / "1030_驗證題.xlsx")

# === QuerySystem 類別 ===
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
        if self.weight_strategy == "increment": return 1 + position*0.5
        if self.weight_strategy == "small_increment": return 1 + position*0.1
        return 1

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
                for mk in matched_keywords:
                    if mk in self.builder.graph and kw in self.builder.graph.neighbors(mk):
                        keyword_scores[kw] += self.special_neighbor
                        special_keyword_hit = True
                        break

        if filter_keywords:
            keyword_scores = {kw: score for kw, score in keyword_scores.items() if kw in self.selected_keywords}

        return keyword_scores, use_weighting, special_keyword_hit

    def match_query_to_standard_question(self, query, normalize=True):
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
                    question_scores[qid] /= (num_keywords ** self.gamma)
        return sorted(question_scores.items(), key=lambda x: -x[1]), use_weighting, special_keyword_hit

# === 單次實驗流程 ===
def run_experiment(mu, neighbor_weight, ac_topn, rerank_threshold, alpha,
                   keyword_threshold, weight_strategy, special_main, special_neighbor,
                   special_standq_ids, special_standq_weight):
    builder = KnowledgeGraphBuilder(mu=mu)
    builder.load_data_from_excel(CURRENT_DIR / "第九版關鍵字.xlsx")
    builder.build_graph()
    system = QuerySystem(builder, neighbor_weight, 1.0, keyword_threshold, weight_strategy,
                         SELECTED_KEYWORDS, special_main, special_neighbor)
    results_all = []

    for idx, row in query_df.iterrows():
        query = str(row.get("text","")).strip()
        true_label = row.get("true_label", None)
        start_time = time.time()
        matched_questions, _, special_keyword_hit = system.match_query_to_standard_question(query)
        ac_ranked = [qid for qid,_ in matched_questions]
        ac_top_qids = ac_ranked[:ac_topn]

        # 強制加入 standq（僅當 special_keyword_hit=True）
        if special_standq_ids and special_keyword_hit:
            for sid in special_standq_ids:
                if sid not in ac_top_qids and sid in label_map:
                    ac_top_qids.append(sid)
                    if sid not in dict(matched_questions):
                        matched_questions.append((sid, 0.0))
                    for i,(qid,score) in enumerate(matched_questions):
                        if qid==sid:
                            matched_questions[i]=(qid, score+special_standq_weight)
                            break

        if not ac_top_qids: continue

        with torch.no_grad():
            query_inputs = tokenizer_cl([query], return_tensors="pt", padding=True, truncation=True).to(device)
            query_emb = cl_model(**query_inputs)
            if isinstance(query_emb,(tuple,list)): query_emb=query_emb[0]
            if query_emb.dim()==3: query_emb=query_emb[:,0,:]

            candidate_texts = [label_map[qid] for qid in ac_top_qids]
            candidate_inputs = tokenizer_cl(candidate_texts, return_tensors="pt", padding=True, truncation=True).to(device)
            candidate_embs = cl_model(**candidate_inputs)
            if isinstance(candidate_embs,(tuple,list)): candidate_embs=candidate_embs[0]
            if candidate_embs.dim()==3: candidate_embs=candidate_embs[:,0,:]

        sims = cosine_similarity(query_emb.cpu().numpy(), candidate_embs.cpu().numpy())[0]
        qid_to_sim = {qid:sim for qid,sim in zip(ac_top_qids, sims)}
        pred_label_cl = max(qid_to_sim,key=qid_to_sim.get)
        top_score_cl = qid_to_sim[pred_label_cl]
        pred_label_final = pred_label_cl
        top_score_final = top_score_cl
        rerank_flag=0

        if top_score_cl < rerank_threshold:
            rerank_flag=1
            sims_array = np.array(sims)
            sims_norm = (sims_array - sims_array.min())/(sims_array.max()-sims_array.min()+1e-8)
            qid_to_sim_norm={qid:sim for qid,sim in zip(ac_top_qids, sims_norm)}
            ac_score_dict={qid:score for qid,score in matched_questions if qid in ac_top_qids}
            max_ac_score=max(ac_score_dict.values()) if ac_score_dict else 1.0
            ac_score_normalized={qid:score/max_ac_score for qid,score in ac_score_dict.items()}
            combined_scores={qid: alpha*qid_to_sim_norm[qid]+(1-alpha)*ac_score_normalized[qid] for qid in ac_top_qids}
            pred_label_final=max(combined_scores,key=combined_scores.get)
            top_score_final=combined_scores[pred_label_final]

        results_all.append({
            "query_id": idx+1,
            "query_text": query,
            "true_standard_question": label_map.get(true_label,"未知"),
            "predicted_standard_question": label_map.get(pred_label_final,"未知"),
            "是否正確": int(pred_label_final==true_label),
            "是否用rerank": rerank_flag,
            "CL過濾TOP1": label_map.get(pred_label_cl,"未知"),
            "CL過濾TOP1分數": top_score_cl,
            "最終預測結果": label_map.get(pred_label_final,"未知"),
            "最終預測分數": top_score_final,
            "耗時(秒)": time.time()-start_time
        })

    out_df = pd.DataFrame(results_all)
    evaluated=len(out_df)
    correct_count=out_df["是否正確"].sum() if evaluated>0 else 0
    accuracy=(correct_count/evaluated) if evaluated>0 else 0
    coverage=(evaluated/len(query_df)) if len(query_df)>0 else 0

    detail_name=(
        f"result_filter{ac_topn}_neighbor{neighbor_weight}_th{rerank_threshold}_alpha{alpha}_mu{mu}"
        f"_kt{keyword_threshold}_{weight_strategy}_sm{special_main}_sn{special_neighbor}"
        f"_sid{','.join(map(str, special_standq_ids or []))}_w{special_standq_weight}.xlsx"
    )
    out_df.to_excel(OUT_DETAIL_DIR / detail_name, index=False)
    return correct_count, accuracy, coverage, evaluated

# === 主程式：中斷續跑 ===
if __name__=="__main__":
    summary_records=[]
    param_combinations=[
        (ac_topn, neighbor_weight, rerank_threshold, alpha, mu,
         keyword_threshold, weight_strategy, special_main, special_neighbor,
         special_standq_ids, special_standq_weight)
        for ac_topn, neighbor_weight, rerank_threshold, alpha, mu, keyword_threshold,
            weight_strategy, special_main, special_neighbor, special_standq_ids, special_standq_weight
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
            SPECIAL_STANDQ_WEIGHT_LIST
        ) if special_main>=special_neighbor
    ]

    existing_details = set(os.listdir(OUT_DETAIL_DIR))

    for combo in tqdm(param_combinations, desc="Resuming experiments", dynamic_ncols=True, ncols=100):
        (ac_topn, neighbor_weight, rerank_threshold, alpha, mu,
         keyword_threshold, weight_strategy, special_main, special_neighbor,
         special_standq_ids, special_standq_weight)=combo

        detail_name=(
            f"result_filter{ac_topn}_neighbor{neighbor_weight}_th{rerank_threshold}_alpha{alpha}_mu{mu}"
            f"_kt{keyword_threshold}_{weight_strategy}_sm{special_main}_sn{special_neighbor}"
            f"_sid{','.join(map(str, special_standq_ids or []))}_w{special_standq_weight}.xlsx"
        )
        detail_path=OUT_DETAIL_DIR/detail_name

        if detail_name in existing_details:
            try:
                out_df=pd.read_excel(detail_path)
                correct_count=out_df["是否正確"].sum()
                evaluated=len(out_df)
                accuracy=correct_count/evaluated if evaluated>0 else 0
                coverage=evaluated/len(query_df) if len(query_df)>0 else 0
                print(f"[SKIP] 已存在：{detail_name}")
            except:
                print(f"[ERROR] 讀取失敗，重新執行 {detail_name}")
                correct_count, accuracy, coverage, evaluated=run_experiment(
                    mu, neighbor_weight, ac_topn, rerank_threshold, alpha,
                    keyword_threshold, weight_strategy, special_main, special_neighbor,
                    special_standq_ids, special_standq_weight
                )
        else:
            print(f"[RUN] 新執行：{detail_name}")
            correct_count, accuracy, coverage, evaluated=run_experiment(
                mu, neighbor_weight, ac_topn, rerank_threshold, alpha,
                keyword_threshold, weight_strategy, special_main, special_neighbor,
                special_standq_ids, special_standq_weight
            )

        summary_records.append({
            "ac_filter_size": ac_topn,
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
            "Correct Count": correct_count,
            "Evaluated": evaluated,
            "Coverage (%)": round(coverage*100,2),
            "Answer_Accuracy (%)": round(accuracy*100,2)
        })

    ac_str="_".join(map(str,AC_TOPN_LIST))
    outfile=f"summary_{datetime_str}_AC{ac_str}.xlsx"
    pd.DataFrame(summary_records).to_excel(CURRENT_DIR/outfile, index=False)
    print(f"\n✅ 完成，輸出 summary 檔案：{outfile}")
