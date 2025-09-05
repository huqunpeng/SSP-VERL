import networkx as nx
import networkx as nx
import numpy as np
import igraph as ig
import networkit as nk
import math
import linkprediction
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import random
import simulate_influence_with_seedsnew
import ssmain1125
import torch
import random
import numpy as np
import networkx as nx
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
from joblib import Parallel, delayed

def read_graph_from_txt():
    G = nx.read_edgelist("google12.txt", nodetype=int)
    for u, v in G.edges():
        G[u][v]["weight"] = 1
    return G
def calculatelocation():
    print("zxynb")
    Gnew=linkprediction.main()
    return Gnew
import cupy as cp
from numba import cuda
def calculate_closeness_centrality(G):
    # 转换边的概率为距离
    for u, v, data in G.edges(data=True):
        data['weight'] = 1 / data['weight']

    # 计算加权closeness centrality
    closeness = nx.closeness_centrality(G, distance='weight')
    return closeness
def calculate_betweenness_centrality(G):
    # 转换边的概率为成本（权重的倒数）
    for u, v, data in G.edges(data=True):
        data['weight'] = 1 / data['weight']  # 将概率倒数作为成本

    # 计算加权betweenness centrality
    betweenness = nx.betweenness_centrality(G, weight='weight')
    return betweenness
def calculate_eigenvector_centrality(G):
    # 确保所有边的权重代表概率或影响强度
    # 这里不需要转换，因为权重已经代表了连接的强度
    centrality = nx.eigenvector_centrality_numpy(G, weight='weight')
    return centrality

@cuda.jit
def update_votes(A, scores, degrees, votes, active_nodes):
    i = cuda.grid(1)
    if i < votes.shape[0]:
        if active_nodes[i]:
            votes[i] = scores[i]  # 初始化自己的得分
        else:
            votes[i] = -1e10
        for j in range(A.shape[1]):
            if A[i, j] > 0 and degrees[j] > 0:
                cuda.atomic.add(votes, j, scores[i] * (A[i, j] / degrees[j]))


@cuda.jit
def decay_neighbors(A, max_idx, scores, decay_factor):
    """
    CUDA 更新被选中节点的邻居影响力
    """
    i = cuda.grid(1)
    if i < A.shape[1] and A[max_idx, i] > 0:
        scores[i] *= decay_factor


def voterank(G, num_seeds):
    # 初始化所有节点的投票权重
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    # 计算特征向量中心性（可能因奇异矩阵或不收敛而失败，使用 try-except）
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.NetworkXError:
        eigenvector_centrality = {node: 0 for node in G.nodes()}

    # 初始化每个节点的投票能力为三种中心性的总和
    votes = {
        node: closeness_centrality[node] + betweenness_centrality[node] + eigenvector_centrality[node]
        for node in G.nodes()
    }

    seeds = []

    for _ in range(num_seeds):
        # 计算每个节点的得票数（根据度在邻居节点的比例进行加权）
        score = {}
        for node in G.nodes():
            if votes[node] > 0:
                total_degree = sum(G.degree(neighbor) for neighbor in G.neighbors(node) if votes[neighbor] > 0)
                if total_degree > 0:
                    score[node] = sum(
                        (G.degree(node) / total_degree) * votes[neighbor]
                        for neighbor in G.neighbors(node) if votes[neighbor] > 0
                    )
                else:
                    score[node] = 0
            else:
                score[node] = 0

        # 选择得票数最高的节点
        selected = max(score, key=score.get)
        seeds.append(selected)

        # 将已选为种子节点的分数置为 0
        votes[selected] = 0

        # 将邻居节点的投票能力衰减为 0.9 倍
        for neighbor in G.neighbors(selected):
            if votes[neighbor] > 0:
                votes[neighbor] *= 0.9

    return seeds


G=read_graph_from_txt()
def build_location_dict(location):
    # 创建一个字典，以节点编号为键，节点属性列表（不包括编号）为值
    location_dict = {}
    for item in location:
        node_id = item[0]  # 节点编号
        node_attributes = item[1:]  # 节点属性
        location_dict[node_id] = node_attributes
    return location_dict

import networkx as nx
import numpy as np
import random
import heapq
from concurrent.futures import ProcessPoolExecutor
import math
from joblib import Parallel, delayed

# 贪心算法：每次选择增益最大的节点
G = read_graph_from_txt()
# 贪心算法：每次选择增益最大的节点
Gnew=calculatelocation()
seed=voterank(G,300)
print(seed)
#seed=[236, 237, 43, 167, 136, 168, 44, 45, 175, 176, 2, 46, 355, 177, 101, 47, 22, 64, 243, 259, 86, 457, 117, 57, 346, 437, 576, 648, 459, 250, 374, 65, 116, 244, 178, 63, 210, 62, 157, 726, 465, 551, 1, 202, 122, 192, 295, 277, 482, 52]
actions = seed.copy()  # 初始化动作空间，复制一份seed以避免修改原始数据



influence_cache = {}
import networkx as nx
import random
import math
import simulate_influence_with_seedsnew
import ssmain1125

# 全局缓存，避免重复计算
import random

import random
import simulate_influence_with_seedsnew
import ssmain1125

import random
import simulate_influence_with_seedsnew
import ssmain1125

import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import networkx as nx
import numpy as np
from tqdm import tqdm
import ray
ray.init()  # 初始化 Ray 运行环境，只需执行一次

@ray.remote
def remote_simulate(G, init_cond, seeds, at, k, random_probs):
    influence = simulate_influence_with_seedsnew.simulate_influence_with_seeds(
        G, init_cond, seeds, at, k, random_probs
    )
    return tuple(seeds), influence

def generate_greedy_trajectory_ray(G, init_cond, candidate_nodes, k, at, random_probs, num_seeds=50):
    selected = []
    trajectory = []
    influence_cache = {}

    for step in range(num_seeds):
        print(f"[step {step}] 当前种子节点数：{len(selected)}")

        base_key = tuple(sorted(selected))
        if base_key not in influence_cache:
            base_infl = simulate_influence_with_seedsnew.simulate_influence_with_seeds(
                G, init_cond, selected, at, k, random_probs
            )
            influence_cache[base_key] = base_infl
        else:
            base_infl = influence_cache[base_key]

        # 构建需要远程仿真的节点组合
        candidates = [v for v in candidate_nodes if v not in selected]
        to_simulate = []
        for v in candidates:
            test_key = tuple(sorted(selected + [v]))
            if test_key not in influence_cache:
                to_simulate.append(v)

        # 并行调度远程仿真
        futures = [
            remote_simulate.remote(G, init_cond, selected + [v], at, k, random_probs)
            for v in to_simulate
        ]
        results = ray.get(futures)

        for seeds_tuple, infl in results:
            influence_cache[tuple(sorted(seeds_tuple))] = infl

        # 计算边际增益
        gains = {}
        for v in candidates:
            test_key = tuple(sorted(selected + [v]))
            if test_key in influence_cache:
                gains[v] = influence_cache[test_key] - base_infl

        if not gains:
            break

        best_node = max(gains, key=gains.get)
        selected.append(best_node)
        trajectory.append((selected.copy(), best_node))

    return trajectory

# ==== 状态向量构造 ====
def build_state_vector(selected, candidate_nodes, num_total_nodes):
    # One-hot编码：当前已选节点集合
    state = np.zeros(num_total_nodes)
    for v in selected:
        state[v] = 1
    return state

# ==== 简单 DQN ====
class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==== DQN 模拟训练 ====
def train_dqn_from_greedy(trajectory, candidate_nodes, num_total_nodes, epochs=30):
    input_dim = num_total_nodes
    hidden_dim = 128
    output_dim = num_total_nodes  # 每个节点一个动作

    dqn = DQN(input_dim, hidden_dim, output_dim)
    optimizer = optim.Adam(dqn.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for state_nodes, action_node in trajectory:
            state_vec = build_state_vector(state_nodes[:-1], candidate_nodes, num_total_nodes)
            state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
            q_values = dqn(state_tensor)

            target = torch.LongTensor([action_node])
            loss = loss_fn(q_values, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")
    torch.save(dqn.state_dict(), 'dqn_model.pth')
    return dqn

# ==== 用训练好的 DQN 预测一组种子节点 ====
def predict_seed_set(dqn, candidate_nodes, num_total_nodes, num_seeds=50):
    selected = []
    for _ in range(num_seeds):
        state_vec = build_state_vector(selected, candidate_nodes, num_total_nodes)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
        with torch.no_grad():
            q_values = dqn(state_tensor).squeeze()

        for v in selected:
            q_values[v] = -float('inf')  # 不选重复节点

        next_node = torch.argmax(q_values).item()
        selected.append(next_node)
    return selected


# ===== 主程序 =====
import time
if __name__ == "__main__":
    G = read_graph_from_txt()

    input_file = "google12random.txt"
    random_probs = []
    with open(input_file, "r") as f:
        for line in f:
            random_probs.append(list(map(float, line.strip().split(","))))

    init_cond = ssmain1125.setup_initial_conditions(G)

    candidate_nodes = seed
    k = 5  # 最多传播轮数
    at = simulate_influence_with_seedsnew.at
    start=time.time()
    print(">>> 生成贪心轨迹...")
    greedy_trajectory = generate_greedy_trajectory_ray(
        G, init_cond, candidate_nodes, k, at, random_probs, num_seeds=50)

    print(">>> 训练 DQN...")
    dqn_model = train_dqn_from_greedy(greedy_trajectory, candidate_nodes, num_total_nodes=len(G.nodes()))

    print(">>> 使用 DQN 选取种子节点...")
    dqn_seeds = predict_seed_set(dqn_model, candidate_nodes, num_total_nodes=len(G.nodes()))

    print("DQN 选取的 50 个种子节点：", dqn_seeds)
    end=time.time()
    print(end-start)





