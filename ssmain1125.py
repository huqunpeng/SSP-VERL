import time

import numpy as np
import networkx as nx
from numba import jit, prange
import concurrent.futures
from scipy.sparse.csgraph import shortest_path
from scipy.sparse.linalg import eigs
from joblib import Parallel, delayed
from scipy.sparse.csgraph import dijkstra
import networkit as nk
import igraph as ig
from scipy.sparse import coo_matrix
from simulate_influence_with_seedsnew import simulate_influence_with_seeds
# 参数配置
global k
at = 10
initialtxt="email-univinitial.txt"

def drawgraph():
    G = nx.read_edgelist("email-univ.txt", nodetype=int)
    return G


def calculatewholeinfluence(location):
    ratio = 2 * (np.sum(location[:, 7] > 0) / len(location)) - 1
    x = 5 * ratio
    # 利用指数表达式实现 tanh(x)
    exp_x = np.exp(x)
    exp_neg_x = np.exp(-x)
    return(exp_x - exp_neg_x) / (exp_x + exp_neg_x)

def fast_degree_centrality_nx(graph):
    # 获取每个节点的度数
    degree = np.array([d for _, d in graph.degree()])
    # 标准化
    return degree / (len(graph))
def fast_closeness_centrality(G):
    ig_graph = ig.Graph.from_networkx(G)

    # 计算 Closeness Centrality
    closeness_centrality = ig_graph.closeness()
    return closeness_centrality
def fast_betweenness_centrality(G):
    nk_graph = nk.nxadapter.nx2nk(G)
    betweenness = nk.centrality.Betweenness(nk_graph, normalized=True)
    betweenness.run()
    return np.array(betweenness.scores())
def fast_eigenvector_centrality(G, max_iter=1000, tol=1e-6):
    n = len(G)
    # 初始化特征向量（全为1）
    centrality = np.ones(n, dtype=float)
    # 邻接矩阵
    adj_matrix = nx.to_scipy_sparse_array(G, format="csr").astype(float)

    for _ in range(max_iter):
        prev_centrality = centrality.copy()
        # 幂迭代
        centrality = adj_matrix @ prev_centrality
        # 归一化
        centrality /= np.linalg.norm(centrality, ord=2)
        # 判断收敛
        if np.linalg.norm(centrality - prev_centrality, ord=1) < tol:
            break

    return centrality


def calculatelocation(G):
    node_ids = np.array(list(sorted(G.nodes())))

    # 计算中心性
    degree_centrality = np.array(list(fast_degree_centrality_nx(G)))
    closeness_centrality = np.array(list(fast_closeness_centrality(G)))
    betweenness_centrality = np.array(list(fast_betweenness_centrality(G)))
    eigenvector_centrality = np.array(list(fast_eigenvector_centrality(G)))

    # 对齐顺序
    node_to_index = {node: idx for idx, node in enumerate(G.nodes())}
    sorted_indices = [node_to_index[node] for node in sorted(G.nodes())]

    degree_centrality = degree_centrality[sorted_indices]
    closeness_centrality = closeness_centrality[sorted_indices]
    betweenness_centrality = betweenness_centrality[sorted_indices]
    eigenvector_centrality = eigenvector_centrality[sorted_indices]

    # 归一化处理
    normalized_degree = 1 / (degree_centrality * (len(node_ids) - 1) + 1)
    epsilon = 1e-6
    normalized_closeness = 0.5 + 0.5 * (1 - (closeness_centrality - closeness_centrality.min()) / (
                closeness_centrality.max() - closeness_centrality.min() + epsilon))
    normalized_betweenness = 0.5 + 0.5 * (1 - (betweenness_centrality - betweenness_centrality.min()) / (
                betweenness_centrality.max() - betweenness_centrality.min() + epsilon))
    normalized_eigenvector = 0.5 + 0.5 * (1 - (eigenvector_centrality - eigenvector_centrality.min()) / (
                eigenvector_centrality.max() - eigenvector_centrality.min() + epsilon))

    data = np.column_stack(
        (node_ids, normalized_degree, normalized_closeness, normalized_betweenness, normalized_eigenvector,
         np.zeros(len(node_ids)), np.zeros(len(node_ids)),
         np.zeros(len(node_ids)), np.zeros(len(node_ids)),
         np.zeros(len(node_ids), dtype=int),
         np.zeros(len(node_ids), dtype=int),
         np.zeros(len(node_ids),),np.zeros(len(node_ids))))
    return data




# 数据分块函数
def chunk_data(rows, cols, distances, chunk_size):
    for start in range(0, len(rows), chunk_size):
        end = start + chunk_size
        yield rows[start:end], cols[start:end], distances[start:end]

def randominitial(numpylocation,initialtxt):
    # 创建随机种子影响力数组
    with open(initialtxt, "r") as file:
        txt_values = [float(line.strip()) for line in file.readlines()]  # 假设每行是一个值

    # 确保 txt 文件中的值数量匹配 numpylocation 的行数
    if len(txt_values) != len(numpylocation):
        raise ValueError("TXT 文件中的值数量必须与 numpylocation 的行数相同！")

    # 将读取的值赋给 numpylocation 的第 7 列
    numpylocation[:, 7] = txt_values


    # 利用向量化操作来设置正负性标记
    numpylocation[:, 9] = np.where(numpylocation[:, 7] >= 1, 1, -1)
    numpylocation[:, 12] = np.where(numpylocation[:, 7] >= 1, 1, -1)
    numpylocation[:, 11] = calculatewholeinfluence(numpylocation)
    # 计算整体影响力并赋值到所有位置的全局影响力字段
    a = (np.sum(numpylocation[:, 7] >0))
    return numpylocation


def setup_initial_conditions(G):
    """ 设置初始条件，仅运行一次 """
    numpylocation = calculatelocation(G)
    numpylocation = randominitial(numpylocation,initialtxt)
    num1=0
    num2=0
    return numpylocation

import numpy as np
import networkx as nx
import concurrent.futures

def main():
    G = drawgraph()
    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()
    k = 5 if num_nodes > 0 else 0
    initial_numpylocation = setup_initial_conditions(G)
#degree degreediscount HDBQ RANDOM
    base_list =[1111, 785, 1104, 871, 1067, 887, 982, 1002, 676, 695, 928, 951, 736, 1115, 1049, 952, 1126, 845, 1040, 1093, 850,
 1107, 1057, 1112, 1122, 1128, 961, 805, 984, 644, 1078, 820, 1131, 668, 944, 880, 1061, 964, 572, 1127, 1085, 995, 740, 685, 774, 631, 622, 1096, 659, 577]
    seed_lists = [base_list[:i] for i in range(5, 51, 5)]
    # 使用并行处理提高性能
    '''with concurrent.futures.ProcessPoolExecutor() as executor:
        # 正确传递参数给并行处理函数
        tasks = [executor.submit(simulate_influence_with_seeds, G, initial_numpylocation, seeds,at,k) for seeds in
                 seed_lists]
        results = [task.result() for task in tasks]'''
    input_file = "google12random.txt"

    random_chances = []  # 用于存储读取的数据
    with open(input_file, "r") as f:
        for line in f:
            # 去掉行尾的换行符并将字符串分割成浮点数列表
            random_chances.append(list(map(float, line.strip().split(","))))
    for seed in seed_lists:
        print(simulate_influence_with_seeds(G,initial_numpylocation,seed,at,k,random_chances))

if __name__ == "__main__":
    start=time.time()
    main()
    # print(1)
    end=time.time()
    # print(end-start)
