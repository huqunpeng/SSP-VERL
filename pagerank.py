import time
import networkx as nx
import numpy as np

def drawgraph():
    G = nx.Graph()
    x, y = np.loadtxt("google12.txt", unpack=True, dtype=int)
    for i in range(len(x)):
        G.add_edge(x[i], y[i])
    return G

def pagerank_seed_selection(G, num_seeds):
    pr = nx.pagerank(G)
    sorted_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    return [node for node, score in sorted_nodes[:num_seeds]]

if __name__ == "__main__":
    G = drawgraph()
    results = {}

    print(">>> PageRank选种子节点耗时对比：")
    for num_seeds in range(5, 55, 5):
        start = time.time()
        for i in range(10):
            seeds = pagerank_seed_selection(G, num_seeds)
        end = time.time()
        elapsed = end - start
        results[num_seeds] = elapsed
        print(f"种子节点数 = {num_seeds}, 耗时 = {elapsed:.4f}s，选中节点: {seeds}")

    print("\n=== 总结表 ===")
    for k, t in results.items():
        print(t)
