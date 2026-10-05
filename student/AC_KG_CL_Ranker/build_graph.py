import pandas as pd
import networkx as nx
import ast
from itertools import combinations


class KnowledgeGraphBuilder:
    def __init__(self, mu=0.8):
        """
        mu: 鄰居重疊比例閾值 (控制 inter edge 是否連邊)
        """
        self.mu = mu
        self.graph = nx.Graph()
        self.standard_questions = {}      # {qid: set(關鍵字)}
        self.keyword_to_questions = {}    # {kw: set(qid)}
        self.qid_to_sentence = {}         # {qid: 標準問句}

    def load_data_from_excel(self, file_path, sheet_name=0):
        """
        從 Excel 讀取資料
        格式: 知識編號 | 標準問句 | 關鍵字 (list字串)
        """
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        for _, row in df.iterrows():
            qid = row["知識編號"]
            sentence = row.get("標準問句", "")
            keywords = row["關鍵字"]
            if isinstance(keywords, str):
                keywords = ast.literal_eval(keywords)
            self.standard_questions[qid] = set(keywords)
            self.qid_to_sentence[qid] = sentence

        # 建立倒排索引
        self.keyword_to_questions = {}
        for qid, kws in self.standard_questions.items():
            for kw in kws:
                self.keyword_to_questions.setdefault(kw, set()).add(qid)

    def build_graph(self):
        """
        建立知識圖:
        - 原則1 (intra edge): 同一句話的關鍵字 fully connected
        - 原則2 (inter edge): 鄰居重疊比例 >= mu
        """
        intra_edges = set()
        inter_edges = set()

        # === 原則1：同一句話的關鍵字 fully connected ===
        for qid, kws in self.standard_questions.items():
            for kw in kws:
                if kw not in self.graph:
                    self.graph.add_node(kw, questions=set())
                self.graph.nodes[kw]["questions"].add(qid)

            for w1, w2 in combinations(sorted(kws), 2):  # 這邊改用sorted排序確保順序一致
                if not self.graph.has_edge(w1, w2):
                    self.graph.add_edge(w1, w2, type="intra", weight=1)
                    intra_edges.add(frozenset([w1, w2]))
                else:
                    self.graph[w1][w2]["weight"] += 1

        # === 原則2：鄰居重疊比例 >= mu ===
        keywords = sorted(self.graph.nodes)  # 固定順序
        inter_edges_to_add = []

        # 先計算所有符合條件的 inter edges，然後最後再一起直接改圖
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                w1, w2 = keywords[i], keywords[j]

                neighbors1 = set(self.graph.neighbors(w1))
                neighbors2 = set(self.graph.neighbors(w2))

                if not neighbors1 or not neighbors2:
                    continue

                inter = len(neighbors1 & neighbors2)
                union = len(neighbors1 | neighbors2)

                if union > 0 and (inter / union) >= self.mu:
                    inter_edges_to_add.append((w1, w2, inter / union))

        # 一次性新增 inter edges，確保順序與結果穩定
        for w1, w2, weight in inter_edges_to_add:
            if not self.graph.has_edge(w1, w2):
                self.graph.add_edge(w1, w2, type="inter", weight=weight)
                inter_edges.add(frozenset([w1, w2]))
            else:
                self.graph[w1][w2]["weight"] += weight
                self.graph[w1][w2]["type"] = "both"
                inter_edges.add(frozenset([w1, w2]))

        # print(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        # print(f"  - Intra edges: {len(intra_edges)}")
        # print(f"  - Inter edges: {len(inter_edges)}")

        return self.graph

# 測試: 看輸出是不是都一樣數量的節點和邊
# kg_builder = KnowledgeGraphBuilder(mu=0.8)
# kg_builder.load_data_from_excel("keywords_output_ll3.xlsx")
# graph = kg_builder.build_graph()