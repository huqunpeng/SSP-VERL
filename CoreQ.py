import time

import networkx as nx
import random
from collections import defaultdict
import torch
import pickle

def k_core_decomposition(G):
    """
    利用 networkx 的 core_number 函数来获取每个节点的核心数。
    返回:
        core_dict: dict，key 为节点，value 为该节点的核数
        C: dict，key 为核心数 core_num，value 为处于该核心数的所有节点列表
        K: int，图中最大核心数
    """
    core_dict = nx.core_number(G)  # 每个节点的核心数
    K = max(core_dict.values())  # 最大核心数
    C = defaultdict(list)  # 按核心数分组
    for node, cnum in core_dict.items():
        C[cnum].append(node)
    return core_dict, C, K


def set_edge_probabilities(G, method='degree_based'):
    """
    为每条边 (u, v) 设置概率 p_{u,v}。
    这里仅给出示例，可根据实际需要进行自定义。
    参数:
        G: networkx.Graph
        method: 选取概率的方式，可自行扩展
    返回:
        p: dict，键为 (u,v) 元组，值为 p_{u,v}
    """
    p = {}
    for u, v in G.edges():
        if method == 'random':
            # 随机生成 0~1 的概率
            prob = random.random()
        elif method == 'degree_based':
            # 简单示例：可根据节点度数做一个反比
            deg_u = G.degree[u]
            deg_v = G.degree[v]
            prob = 1.0 / (deg_u + deg_v) if (deg_u + deg_v) > 0 else 0
        else:
            # 默认给个固定值
            prob = 0.5

        p[(u, v)] = prob
        p[(v, u)] = prob  # 无向图可令 p_{u,v} = p_{v,u}
    return p


def compute_node_score(G, p, node, L_of_node):
    """
    计算单个节点的打分 Sc(node) = sum(1 - p_{u,node} for u in L(node)).
    参数:
        G: networkx.Graph
        p: dict, 边概率字典 {(u,v): p_uv}
        node: 要计算的节点
        L_of_node: 给定节点 node 对应的节点集合 L(node)
    返回:
        score: 该节点的打分值
    """
    score = 0.0
    for u in L_of_node:
        # 如果 (u, node) 是一条边，则获取其概率；否则视为 p_{u,node} = 0
        if (u, node) in p:
            score += (1 - p[(u, node)])
        else:
            # 如果没有边，可以视为 p_{u,node} = 0，则 (1 - 0) = 1
            # 具体是否这样处理要看论文或算法的定义
            score += 1
    return score


def candidate_seed_node_set_selection(G, l, p=0.5):
    """
    实现“算法 1: 候选种子节点集合选取”，在下述前提下：
      1) L(w) 为与 w 同核数(core number)的邻居集合；
      2) 每条边 (u,v) 的概率 p_{u,v} = p（一个常数）。

    参数：
        G : networkx.Graph
            输入无向图
        l : int
            要选取的种子节点数量
        p : float
            每条边的相同传播概率，默认 0.5

    返回：
        R : list
            打分最高的前 l 个节点
    """
    # 1) 计算图中每个节点的核心数 (core number)
    core_dict = nx.core_number(G)  # 返回 {node: core_num}

    # 2) 计算每个节点 w 的打分 Sc(w)
    #    Sc(w) = ∑ (1 - p_{u,w})，其中 u ∈ L(w)
    #    本题中 p_{u,w} = p 为常数
    #    L(w) = { u | u 是 w 的邻居，且 core_dict[u] == core_dict[w] }
    Sc = {}
    for w in G.nodes():
        # 找到与 w 同核心数的邻居
        neighbors_same_core = [
            u for u in G.neighbors(w)
            if core_dict[u] == core_dict[w]
        ]
        # 计算打分
        # 每个 u 都贡献 (1 - p)
        Sc[w] = len(neighbors_same_core) * (1 - p)

    # 3) 按照打分从大到小排序，取前 l 个节点
    sorted_nodes = sorted(Sc.keys(), key=lambda x: Sc[x], reverse=True)
    R = sorted_nodes[:l]
    return R

def read_graph_from_txt():
    G = nx.read_edgelist("google12.txt", nodetype=int)
    for u, v in G.edges():
        G[u][v]["weight"] = 1
    return G

def epsilon_greedy(Q, state, actions, epsilon=0.1):
    if not actions:
        return None
    if random.random() < epsilon:
        return random.choice(actions)
    else:
        return max(actions, key=lambda action: Q.get((state, action), 0))


def compute_P_w(G, S, w):
    neighbors = set(G.neighbors(w)) & S
    product_term = 1.0
    for v in neighbors:
        product_term *= (1 - G[w][v]['weight'])
    return 1 - product_term


def compute_D_w(G, w):
    return sum(G[w][v]['weight'] for v in G.neighbors(w))


def compute_sigma_prime(G, S):
    if not S:  # 避免 S 为空时 `set.union()` 报错
        return 0

    neighbor_sets = [set(G.neighbors(s)) for s in S]
    all_neighbors = set().union(*neighbor_sets) if neighbor_sets else set()

    sigma_prime = 0
    for w in all_neighbors:
        P_w = compute_P_w(G, S, w)
        D_w = compute_D_w(G, w)
        sigma_prime += P_w * D_w

    return sigma_prime



def q_learning(G, candidate_seeds, episodes=50, alpha=0.5, gamma=0.7, epsilon=0.1):
    Q = {}  # Q-table
    num_seeds = 50
    for _ in range(episodes):
        state = frozenset()
        available_actions = candidate_seeds.copy()
        prev_sigma = compute_sigma_prime(G, state)

        for _ in range(num_seeds):
            if not available_actions:
                break

            action = epsilon_greedy(Q, state, available_actions, epsilon)
            if action is None:
                break

            available_actions.remove(action)
            new_state = state | {action}
            curr_sigma = compute_sigma_prime(G, new_state)

            marginal_gain = curr_sigma - prev_sigma
            reward = marginal_gain

            max_next_q = max([Q.get((new_state, next_action), 0) for next_action in available_actions], default=0)
            Q[(state, action)] = Q.get((state, action), 0) + alpha * (
                    reward + gamma * max_next_q - Q.get((state, action), 0)
            )

            prev_sigma = curr_sigma
            state = new_state
    return Q

# 根据学习结果选择最优种子
def select_best_seeds(Q, candidate_seeds, num_seeds=50):
    state = frozenset()
    best_seeds = []
    available_actions = candidate_seeds.copy()
    seed_scores = {}

    while len(best_seeds) < num_seeds and available_actions:
        next_seed = max(available_actions, key=lambda x: Q.get((state, x), 0))
        best_seeds.append(next_seed)
        seed_scores[next_seed] = Q.get((state, next_seed), 0)
        state = state | {next_seed}
        available_actions.remove(next_seed)

    best_seeds.sort(key=lambda x: seed_scores[x], reverse=True)

    return best_seeds
if __name__ == "__main__":
    # 构造图
    start=time.time()
    G = read_graph_from_txt()

    # 选取候选种子节点数量 l
    candidate_seeds = candidate_seed_node_set_selection(G, 300)
    print("Candidate seed nodes:", candidate_seeds)

    Q_loaded = pickle.load(open("q_table.pkl", "rb"))
    selected_seeds = select_best_seeds(Q_loaded, candidate_seeds, num_seeds=15)
    end=time.time()
    print(end-start)
