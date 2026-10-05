import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
import os
import sys
from pathlib import Path # 
import numpy as np
import time

# ---
# 1. 
# 
# 
CURRENT_DIR = Path(__file__).resolve().parent # 
TRAIN_MODEL_DIR = CURRENT_DIR / "Train_model"
sys.path.append(str(TRAIN_MODEL_DIR))

try:
    from Train_model.model_train import CLBERT, build_model
    from Train_model.hyperparameter import PARAM
except ImportError as e:
    print(f"ERROR:  {TRAIN_MODEL_DIR}")
    print(f": {e}")
    sys.exit(1)

MODEL_NAME = 'CL_student_2026-02-22-layer12' # 
MODEL_BASE_DIR = CURRENT_DIR / 'model' # 
GALLERY_FILE = CURRENT_DIR / '編號對應標準問句_增加類別251.xlsx' # 
TEST_FILE = CURRENT_DIR / 'user_query.xlsx' # 

# 
# 
GALLERY_COLUMN = '標準問句' # 
TEST_QUERY_COLUMN = 'text' # 
TEST_TRUTH_COLUMN = 'true_class' # 
# ---

@torch.no_grad()
def get_embeddings(model, tokenizer, texts, device, max_len=30):
    """
    
    """
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    embeddings = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        token_type_ids=inputs['token_type_ids']
    )
    return F.normalize(embeddings, p=2, dim=1)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. 
    model_path = MODEL_BASE_DIR / MODEL_NAME # 
    print(f"Loading model from: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Warning: Could not load tokenizer from {model_path}. Falling back to default: {PARAM['config']}")
        tokenizer = AutoTokenizer.from_pretrained(PARAM['config'])

    model_args = PARAM.copy()
    model_args['model_type'] = 'CLBERT' # 
    model = CLBERT.from_pretrained(model_path, args=model_args).to(device)
    model.eval()

    # --- 4. 
    try:
        gallery_df = pd.read_excel(GALLERY_FILE)
        standard_questions_list = gallery_df[GALLERY_COLUMN].tolist()
        print(f"Loaded {len(standard_questions_list)} standard questions (Gallery) from {GALLERY_FILE}.")
    except Exception as e:
        print(f"ERROR:  {GALLERY_FILE}.: {e}")
        return
    gallery_embeddings = get_embeddings(
            model, 
            tokenizer, 
            standard_questions_list, 
            device, 
            PARAM['max_len']
        )

    # --- 5. 
    print("Generating embeddings for test set (Query)...")
        # =========================
    # Timing accumulators
    # =========================
    t_sum_offline = 0.0   # A: 不含 gallery 進 encoder（gallery embeddings 已離線）
    t_sum_onthefly = 0.0  # B: 含 gallery 進 encoder（每次都重算 gallery embeddings）
    t_cnt = 0
    try:
        test_df = pd.read_excel(TEST_FILE)
        test_texts = test_df[TEST_QUERY_COLUMN].tolist()
        true_std_questions = test_df[TEST_TRUTH_COLUMN].tolist() # 
        print(f"Loaded {len(test_texts)} test items from {TEST_FILE}.")
    except Exception as e:
        print(f"ERROR: {TEST_FILE}.: {e}")
        return
    # --- 6. 相似度計算 ---

        # ======================================================
    # CL-only inference with mean ms/query
    # A) Offline: gallery embeddings 已預算（你現在的設定）
    # B) On-the-fly: 每個 query 都把 gallery texts 再丟進 encoder（純對照）
    # ======================================================
    print("Performing similarity search (per-query timing)...")
    K = 5

    top_k_indices = []  # list of np.array shape [K]

    for q in tqdm(test_texts, desc="CL inference", ncols=100):
        # ---------- A: offline (gallery 已預算) ----------
        t0 = time.perf_counter()

        q_emb = get_embeddings(model, tokenizer, [q], device, PARAM['max_len'])  # [1,H]
        sim = F.cosine_similarity(q_emb, gallery_embeddings, dim=1)              # [G]
        vals, idx = torch.topk(sim, k=K, dim=0)

        t1 = time.perf_counter()

        # ---------- B: on-the-fly (gallery 現算) ----------
        # 只計時，不影響你的 topK 結果（topK 仍用 offline 的 idx）
        t2 = time.perf_counter()

        _gallery_emb_tmp = get_embeddings(model, tokenizer, standard_questions_list, device, PARAM['max_len'])
        _ = _gallery_emb_tmp  # 避免 lint 抱怨

        t3 = time.perf_counter()

        # 累積時間
        t_sum_offline += (t1 - t0)
        t_sum_onthefly += (t3 - t2)
        t_cnt += 1

        top_k_indices.append(idx.cpu().numpy())

    top_k_indices = np.stack(top_k_indices, axis=0)  # [N,K]

    # ---------- timing report ----------
    if t_cnt > 0:
        mean_ms_A = (t_sum_offline / t_cnt) * 1000.0
        mean_ms_B = (t_sum_onthefly / t_cnt) * 1000.0
        print("\n=== 推論時間 (mean ms/query) ===")
        print(f"A) 不含 centers 進 encoder（Offline pre-computation）: {mean_ms_A:.3f} ms/query (N={t_cnt})")
        print(f"B) 含 centers 進 encoder（On-the-fly encoding）:      {mean_ms_B:.3f} ms/query (N={t_cnt})")

    # --- 7. Metrics: Top-K Accuracy, MRR, nDCG ---
    correct_count_k1 = 0
    correct_count_k3 = 0
    correct_count_k5 = 0

    mrr_total = 0.0
    ndcg1_total = 0.0
    ndcg3_total = 0.0
    ndcg5_total = 0.0

    total_count = len(true_std_questions)

    for i in range(total_count):
        truth = true_std_questions[i]
        retrieved_indices = top_k_indices[i]
        retrieved_strings = [standard_questions_list[idx] for idx in retrieved_indices]

        # ----- Top-K Accuracy -----
        if truth == retrieved_strings[0]:
            correct_count_k1 += 1
        if truth in retrieved_strings[:3]:
            correct_count_k3 += 1
        if truth in retrieved_strings:
            correct_count_k5 += 1

        # ----- MRR -----
        if truth in retrieved_strings:
            rank = retrieved_strings.index(truth) + 1  # rank 從 1 開始
            mrr_total += 1.0 / rank

        # ----- nDCG -----
        # relevance: 1 = correct, 0 = incorrect
        rel = [1 if s == truth else 0 for s in retrieved_strings]

        # DCG@K = Σ (rel_i / log2(i+2))
        def dcg(rels, k):
            return sum(rel / (np.log2(idx + 2)) for idx, rel in enumerate(rels[:k]))

        # IDCG@K = 排序後的理想 DCG
        def idcg(k):
            return 1.0  # 因為只有 1 個 relevant，因此 DCG 理想值永遠是 1/log2(1+1)=1

        ndcg1_total += dcg(rel, 1) / idcg(1)
        ndcg3_total += dcg(rel, 3) / idcg(3)
        ndcg5_total += dcg(rel, 5) / idcg(5)

    # --- Final Metrics ---
    accuracy_k1 = correct_count_k1 / total_count * 100
    accuracy_k3 = correct_count_k3 / total_count * 100
    accuracy_k5 = correct_count_k5 / total_count * 100

    mrr = mrr_total / total_count
    ndcg1 = ndcg1_total / total_count
    ndcg3 = ndcg3_total / total_count
    ndcg5 = ndcg5_total / total_count

    print("---Evaluation Complete---")
    print(f"Total Test Items: {total_count}")
    print(f"Top-1 Accuracy: {accuracy_k1:.2f}%")
    print(f"Top-3 Accuracy: {accuracy_k3:.2f}%")
    print(f"Top-5 Accuracy: {accuracy_k5:.2f}%")

    print(f"MRR: {mrr:.4f}")
    print(f"nDCG@1: {ndcg1:.4f}")
    print(f"nDCG@3: {ndcg3:.4f}")
    print(f"nDCG@5: {ndcg5:.4f}")


if __name__ == "__main__":
    main()