import networkx as nx
import  time
def get_neighbors(G, node, depth=2):
    """获取指定节点的深度邻居"""
    neighbors = {1: set(G.neighbors(node))}
    for d in range(2, depth + 1):
        neighbors[d] = set()
        for n in neighbors[d - 1]:
            neighbors[d].update(G.neighbors(n))
        neighbors[d] -= neighbors[d - 1]  # 避免重复
    return neighbors

def im_ncvr_algorithm(G, k=50):
    """基于NCVR（核数中心性+投票机制）选择k个影响力最大的种子节点"""
    seeds = []
    S_ = []
    S = dict()
    Va = dict()
    NC = dict()

    coreness = nx.core_number(G)  # 计算核数核心性
    min_core = min(coreness.values())
    max_core = max(coreness.values())

    # 归一化核心性
    for node in G.nodes():
        NC[node] = (coreness[node] - min_core) / (max_core - min_core) if max_core > min_core else 0
        S[node] = 0
        Va[node] = 1

    avg_degree = sum(dict(G.degree()).values()) / len(G)  # 计算平均度数
    theta = 0.5
    j = 0

    while j < k:
        # 计算投票分数
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            S[node] = sum([Va[i] * NC[i] * (1 - theta) + Va[i] * theta for i in neighbors])

        # 选取投票分数最高的节点作为种子
        for node in seeds:
            S[node] = float('-inf')  # 避免重复选择
        seed = max(S, key=S.get)

        # 记录选中的种子
        S_.append(S[seed])
        seeds.append(seed)
        j += 1

        # 更新邻居的 Va 值
        Va[seed] = 0
        neighbor2 = get_neighbors(G, seed, 2)
        for d in [1, 2]:
            for node in neighbor2[d]:
                delta = 1 / (avg_degree * d)
                Va[node] = max(Va[node] - delta, 0)

    return seeds

def drawgraph():
    """读取图数据"""
    G = nx.read_edgelist("google12.txt", nodetype=int)
    return G

# 加载图并选取影响力最大的 k 个种子节点
start=time.time()
G = drawgraph()
seed_nodes = im_ncvr_algorithm(G, k=5)
end=time.time()
# 输出选出的种子节点
print("选出的种子节点:", seed_nodes)
print(end-start)
