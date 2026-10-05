import os
import pickle
import pandas as pd
import networkx as nx
import ast
from itertools import combinations


class KnowledgeGraphBuilder:
    def __init__(self, mu=0.8):
        self.mu = mu
        self.graph = nx.Graph()
        self.standard_questions = {}
        self.keyword_to_questions = {}
        self.qid_to_sentence = {}

    def load_data_from_excel(self, file_path, sheet_name=0):
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        for _, row in df.iterrows():
            qid = row["知識編號"]
            sentence = row.get("標準問句", "")
            keywords = row["關鍵字"]
            if isinstance(keywords, str):
                keywords = ast.literal_eval(keywords)
            self.standard_questions[qid] = set(keywords)
            self.qid_to_sentence[qid] = sentence

        self.keyword_to_questions = {}
        for qid, kws in self.standard_questions.items():
            for kw in kws:
                self.keyword_to_questions.setdefault(kw, set()).add(qid)

    def build_graph(self):
        intra_edges = set()
        inter_edges = set()

        for qid, kws in self.standard_questions.items():
            for kw in kws:
                if kw not in self.graph:
                    self.graph.add_node(kw, questions=set())
                self.graph.nodes[kw]["questions"].add(qid)

            for w1, w2 in combinations(sorted(kws), 2):
                if not self.graph.has_edge(w1, w2):
                    self.graph.add_edge(w1, w2, type="intra", weight=1)
                    intra_edges.add(frozenset([w1, w2]))
                else:
                    self.graph[w1][w2]["weight"] += 1

        keywords = sorted(self.graph.nodes)
        inter_edges_to_add = []

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

        for w1, w2, weight in inter_edges_to_add:
            if not self.graph.has_edge(w1, w2):
                self.graph.add_edge(w1, w2, type="inter", weight=weight)
                inter_edges.add(frozenset([w1, w2]))
            else:
                self.graph[w1][w2]["weight"] += weight
                self.graph[w1][w2]["type"] = "both"
                inter_edges.add(frozenset([w1, w2]))

        return self.graph

    def save_graph(self, file_path):
        # 儲存 graph + keyword_to_questions + standard_questions
        data_to_save = {
            "graph": self.graph,
            "standard_questions": self.standard_questions,
            "keyword_to_questions": self.keyword_to_questions
        }
        with open(file_path, "wb") as f:
            pickle.dump(data_to_save, f)
        print(f"Graph 已儲存到 {file_path}")

    def load_graph(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            self.graph = data["graph"]
            self.standard_questions = data["standard_questions"]
            self.keyword_to_questions = data["keyword_to_questions"]
            print(f"Graph 已從 {file_path} 載入")
        else:
            raise FileNotFoundError(f"{file_path} 不存在")


if __name__ == "__main__":
    kg_builder = KnowledgeGraphBuilder(mu=0.8)
    kg_builder.load_data_from_excel("標準問句關鍵字.xlsx")
    kg_builder.build_graph()
    kg_builder.save_graph("knowledge_graph.pkl")
