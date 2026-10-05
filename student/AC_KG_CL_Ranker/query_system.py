import ahocorasick
from collections import defaultdict
import pandas as pd
import ast

class QuerySystem:
    def __init__(self, standard_sentences, keywords_mapping, neighbor_weight=0.1, gamma=1):
        self.standard_sentences = standard_sentences
        self.keywords_mapping = keywords_mapping
        self.neighbor_weight = neighbor_weight
        self.gamma = gamma

        # --- Aho-Corasick 建立 ---
        self.AC = ahocorasick.Automaton()
        for kw in keywords_mapping.keys():
            self.AC.add_word(kw, kw)
        self.AC.make_automaton()

    def match_query_to_standard_question(self, query, normalize=True, top_k=50):
        matched_keywords = set()
        for _, kw in self.AC.iter(query):
            matched_keywords.add(kw)

        if not matched_keywords:
            return pd.DataFrame(), pd.DataFrame()

        candidate_qs = defaultdict(float)

        # --- 直接匹配 (gamma 分數) ---
        for kw in matched_keywords:
            for qid in self.keywords_mapping.get(kw, []):
                candidate_qs[qid] += self.gamma

        # --- 鄰居匹配 (neighbor_weight 分數) ---
        for kw in matched_keywords:
            for neighbor in self.keywords_mapping.keys():
                if neighbor != kw:
                    for qid in self.keywords_mapping.get(neighbor, []):
                        candidate_qs[qid] += self.neighbor_weight

        # 排序
        ranked = sorted(candidate_qs.items(), key=lambda x: -x[1])
        if top_k:
            ranked = ranked[:top_k]

        # --- 關鍵字表 ---
        kw_with_scores = pd.DataFrame(
            [{"keyword": kw, "score": self.gamma} for kw in matched_keywords]
        )

        # --- 標準問句表 ---
        matched_questions = pd.DataFrame(
            [
                {
                    "qid": qid,
                    "標準問句": self.standard_sentences[qid],
                    "score": score,
                    "rank": rank + 1,
                }
                for rank, (qid, score) in enumerate(ranked)
            ]
        )

        if normalize and not matched_questions.empty:
            total = matched_questions["score"].sum()
            if total > 0:
                matched_questions["score"] = matched_questions["score"] / total

        return kw_with_scores, matched_questions


