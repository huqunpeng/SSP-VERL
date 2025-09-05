import time
import networkx as nx

def kshell_im_algorithm(G, k_shell_values, k=5):
    """基于已计算的 k-shell 值选择 k 个影响力最大的种子节点"""
    sorted_nodes = sorted(G.nodes(),
                          key=lambda node: (k_shell_values[node], G.degree(node)),
                          reverse=True)
    return sorted_nodes[:k]

def drawgraph():
    """读取图数据"""
    G = nx.read_edgelist("google12.txt", nodetype=int)
    return G

G = drawgraph()

# 提前计算一次 k-shell 值，避免重复计算
k_shell_values = nx.core_number(G)

# 分别计算不同k值下的运行时间
time_results = {}

for k in range(5, 55, 5):
    start = time.time()
    for _ in range(100):
        seed_nodes = kshell_im_algorithm(G, k_shell_values, k=k)
    end = time.time()
    total_time = end - start
    time_results[k] = total_time
    print(f"k={k}, 耗时={total_time:.4f}s")

print("\n所有耗时结果：")
for k, t in time_results.items():
    print(t)
