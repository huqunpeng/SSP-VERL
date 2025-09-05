import math
import random
import networkx as nx
from collections import defaultdict
from tqdm import tqdm


class TIMPlusSimple:
    def __init__(self, graph: nx.Graph, budget: int, p: float = 0.01, epsilon: float = 0.2, l: int = 1):
        self.graph = graph.to_directed()
        self.budget = budget
        self.n = len(self.graph.nodes)
        self.m = len(self.graph.edges)
        self.p = p
        self.epsilon = epsilon
        self.l = l
        self.epsilon_prime = 5 * ((l * (epsilon ** 2)) / (budget + l)) ** (1 / 3)
        self.lambda_val = (2 + self.epsilon_prime) * l * self.n * math.log(self.n) * (1 / self.epsilon_prime ** 2)
        self.rr_sets = []
        self.theta = None
        self.kpt = None
        self.occurrences = None

    def generate_rr_ic(self, source):
        rr = set()
        queue = [source]
        rr.add(source)
        while queue:
            current = queue.pop()
            for neighbor in self.graph.predecessors(current):
                if neighbor not in rr and random.random() < self.p:
                    rr.add(neighbor)
                    queue.append(neighbor)
        return rr

    def build_rr_sets(self, num, show_progress=False):
        rr_sets = []
        progress = tqdm(total=num, desc="Building RR sets") if show_progress else None
        for _ in range(num):
            v = random.choice(list(self.graph.nodes))
            rr_sets.append(self.generate_rr_ic(v))
            if progress:
                progress.update(1)
        if progress:
            progress.close()
        return rr_sets

    def kpt_estimation(self):
        for i in range(1, math.floor(math.log2(self.n))):
            c_i = math.floor((6 * self.l * math.log(self.n) + 6 * math.log(math.log2(self.n))) * (2 ** i))
            rr_sets = self.build_rr_sets(c_i)
            sum_kappa = 0
            for rr in rr_sets:
                in_deg = sum([self.graph.in_degree(v) for v in rr])
                kappa = 1 - (1 - in_deg / self.m) ** self.budget
                sum_kappa += kappa
            if (sum_kappa / c_i) > 1 / (2 ** i):
                return self.n * (sum_kappa / (2 * c_i))
        return 1

    def kpt_refinement(self, rr_sets):
        theta = math.floor(self.lambda_val / self.kpt)
        occurrences = defaultdict(set)
        for i, rr in enumerate(rr_sets):
            for node in rr:
                occurrences[node].add(i)

        selected = []
        covered = set()
        for _ in range(self.budget):
            node = max(occurrences.items(), key=lambda x: len(x[1]))[0]
            selected.append(node)
            covered |= occurrences[node]
            occurrences.pop(node)
            for v in occurrences:
                occurrences[v] -= covered

        remaining = theta - len(rr_sets)
        if remaining > 0:
            rr_sets += self.build_rr_sets(remaining)
        f = sum(any(v in rr for v in selected) for rr in rr_sets) / len(rr_sets)
        kpt_prime = f * (self.n / (1 + self.epsilon_prime))
        return max(self.kpt, kpt_prime), rr_sets

    def node_selection(self, rr_sets):
        occurrences = defaultdict(set)
        for i, rr in enumerate(rr_sets):
            for v in rr:
                occurrences[v].add(i)
        selected = []
        for _ in range(self.budget):
            node = max(occurrences.items(), key=lambda x: len(x[1]))[0]
            selected.append(node)
            covered = occurrences[node]
            del occurrences[node]
            for v in occurrences:
                occurrences[v] -= covered
        return selected

    def run(self):
        self.kpt = self.kpt_estimation()
        rr_sets = self.build_rr_sets(math.floor(self.lambda_val / self.kpt), show_progress=True)
        self.kpt, rr_sets = self.kpt_refinement(rr_sets)
        self.theta = math.floor(self.lambda_val / self.kpt)
        if len(rr_sets) < self.theta:
            rr_sets += self.build_rr_sets(self.theta - len(rr_sets), show_progress=True)
        seeds = self.node_selection(rr_sets)
        return seeds


# ==========================
# 主程序入口
# ==========================
def load_graph(path):
    G = nx.Graph()
    with open(path, 'r') as f:
        for line in f:
            u, v = map(int, line.strip().split())
            G.add_edge(u, v)
    return G

import time
if __name__ == "__main__":
    # 修改为你的 celegans.txt 路径
    graph = load_graph("google12.txt")
    print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    start=time.time()
    model = TIMPlusSimple(graph, budget=5, p=0.01)
    seeds = model.run()
    end=time.time()
    print(end-start)
    print("Selected seed nodes:", seeds)
