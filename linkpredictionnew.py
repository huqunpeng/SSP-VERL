import networkx as nx
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import from_networkx, negative_sampling
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
# 读取图
def read_graph_from_txt():
    G = nx.read_edgelist("facebook_combined.txt", nodetype=int)
    return G

# 生成随机初始化的 32 维特征向量
def generate_random_features(G, feature_dim=32):
    np.random.seed(42)  # 固定随机种子，确保可复现
    feature_matrix = np.random.rand(len(G.nodes()), feature_dim)
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    return feature_matrix

# 定义 GraphSAGE 模型
class GraphSAGE(torch.nn.Module):
    def __init__(self, feature_size, hidden_size, embedding_size):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(feature_size, hidden_size)
        self.conv2 = SAGEConv(hidden_size, hidden_size)  # 第二层隐藏层
        self.conv3 = SAGEConv(hidden_size, embedding_size)  # 第三层嵌入层

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)  # 输出嵌入向量
        return x

# 定义 MLP 进行边概率预测
class EdgePredictor(torch.nn.Module):
    def __init__(self, embedding_size):
        super(EdgePredictor, self).__init__()
        self.fc1 = torch.nn.Linear(2 * embedding_size, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, src_emb, dst_emb):
        x = torch.cat([src_emb, dst_emb], dim=1)  # 拼接两个节点的嵌入
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # 预测概率
        return x

# 训练函数
def train(model, predictor, data, optimizer, criterion, device):
    model.train()
    predictor.train()

    optimizer.zero_grad()
    embeddings = model(data)

    # 生成负样本（不存在的边）
    pos_edge_index = data.edge_index
    neg_edge_index = negative_sampling(edge_index=pos_edge_index, num_nodes=data.num_nodes, num_neg_samples=pos_edge_index.size(1))

    # 正样本（存在的边）
    src_pos, dst_pos = pos_edge_index
    pos_pred = predictor(embeddings[src_pos], embeddings[dst_pos])

    # 负样本（不存在的边）
    src_neg, dst_neg = neg_edge_index
    neg_pred = predictor(embeddings[src_neg], embeddings[dst_neg])

    # 计算损失
    labels = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)], dim=0)
    loss = criterion(torch.cat([pos_pred, neg_pred], dim=0), labels)

    loss.backward()
    optimizer.step()
    return loss.item()

# 评估函数
def evaluate(model, predictor, data, device):
    model.eval()
    predictor.eval()
    with torch.no_grad():
        embeddings = model(data)

        # 生成负样本
        pos_edge_index = data.edge_index
        neg_edge_index = negative_sampling(edge_index=pos_edge_index, num_nodes=data.num_nodes, num_neg_samples=pos_edge_index.size(1))

        # 计算预测概率
        src_pos, dst_pos = pos_edge_index
        pos_pred = predictor(embeddings[src_pos], embeddings[dst_pos])

        src_neg, dst_neg = neg_edge_index
        neg_pred = predictor(embeddings[src_neg], embeddings[dst_neg])

        # 计算 AUC
        from sklearn.metrics import roc_auc_score
        labels = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
        scores = torch.cat([pos_pred, neg_pred], dim=0).cpu().numpy()
        auc = roc_auc_score(labels, scores)

    return auc

# 预测缺失边的概率
def predict_edges(model, predictor, data, batch_size=8192, similarity_threshold=0.8):
    model.eval()
    predictor.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    with torch.no_grad():
        embeddings = model(data)  # 获取所有节点嵌入

        # 使用余弦相似度计算节点嵌入的相似性
        similarities = cosine_similarity(embeddings.cpu().numpy())

        # 选择相似度较高的节点对作为候选边
        candidate_pairs = []
        for i in range(len(similarities)):
            for j in range(i + 1, len(similarities)):  # 只考虑 (i, j) 和 (j, i) 不重复
                if similarities[i, j] > similarity_threshold:
                    candidate_pairs.append((i, j))

        # 将候选边转换为张量
        candidate_tensor = torch.tensor(candidate_pairs, dtype=torch.long, device=device)

        edge_probs = []
        for batch_start in range(0, len(candidate_tensor), batch_size):
            batch = candidate_tensor[batch_start:batch_start + batch_size]
            src, dst = batch.T
            probs = predictor(embeddings[src], embeddings[dst]).squeeze().cpu().numpy()

            edge_probs.append((src[i].item(), dst[i].item(), probs[i]))

    return edge_probs
def calculatenewlocation(G, initialtxt):
    # 获取节点编号，并按编号排序
    node_ids = np.array(sorted(G.nodes()))

    # 直接计算中心性并缓存
    degree = np.array(list(calculate_weighted_degree(G).values()))
    degree_centrality = np.array(list(calculate_weighted_degree_centrality(G).values()))
    closeness_centrality = np.array(list(calculate_closeness_centrality(G).values()))
    betweenness_centrality = np.array(list(calculate_betweenness_centrality(G).values()))
    eigenvector_centrality = np.array(list(calculate_eigenvector_centrality(G).values()))
    k_shell_values = np.array(list(calculate_k_shell(G).values()))

    # 节点编号与中心性指标对齐
    node_to_index = {node: idx for idx, node in enumerate(G.nodes())}
    sorted_indices = [node_to_index[node] for node in node_ids]  # 按照排序的节点编号索引

    degree = degree[sorted_indices]
    degree_centrality = degree_centrality[sorted_indices]
    closeness_centrality = closeness_centrality[sorted_indices]
    betweenness_centrality = betweenness_centrality[sorted_indices]
    eigenvector_centrality = eigenvector_centrality[sorted_indices]
    k_shell_values = k_shell_values[sorted_indices]

    # 从文件中读取初始影响力
    with open(initialtxt, "r") as file:
        txt_values = [float(line.strip()) for line in file.readlines()]
    txt_values = np.array(txt_values)

    if len(txt_values) != len(node_ids):
        raise ValueError("节点数量和输入文件行数不一致！")

    # 初始化正向邻居的计数和加权影响力和
    positive_count = np.zeros(len(node_ids))
    positive_sum = np.zeros(len(node_ids))

    # 计算每个节点的周围正向节点数量和正向和
    for i, node in enumerate(node_ids):
        # 初始化
        positive_prob_sum = 0.0
        sum_positive_influences = 0.0

        # 遍历邻居
        for neighbor in G.neighbors(node):
            edge_weight = G[node][neighbor]['weight']
            neighbor_index = node_to_index[neighbor]  # 确保索引一致

            # 如果邻居是正向的
            if txt_values[neighbor_index] > 0:
                positive_prob_sum += edge_weight
                sum_positive_influences += txt_values[neighbor_index] * edge_weight

        # 更新正向邻居的数量和加权影响力和
        positive_count[i] = positive_prob_sum
        positive_sum[i] = sum_positive_influences

    # 数据存储：编号、度、中心性、k-shell、影响力等
    data = np.column_stack(
        (node_ids, degree, degree_centrality, closeness_centrality, betweenness_centrality, eigenvector_centrality,
         k_shell_values, txt_values, np.zeros(len(node_ids)), positive_count, positive_sum,
         np.zeros(len(node_ids)), np.zeros(len(node_ids)), np.zeros(len(node_ids), dtype=int))
    )

    return data

def update_G(G,edge_probs):
    for u, v in G.edges():
        G[u][v]['weight'] = 1
    for src, dst, prob in edge_probs:
        G.add_edge(src, dst, weight=prob)
def calculate_weighted_degree_centrality(G):
    # 计算每个节点的加权度
    weighted_degrees = {node: sum(data['weight'] for _, _, data in G.edges(node, data=True)) for node in G.nodes()}

    # 计算可能的最大加权度
    max_possible_weighted_degree = (len(G.nodes()) - 1)  # 假设最大权重为1且完全连接

    # 计算加权度中心性
    weighted_degree_centrality = {node: deg / max_possible_weighted_degree for node, deg in weighted_degrees.items()}

    return weighted_degree_centrality
def calculate_weighted_degree(G):
    weighted_degrees = {}
    for node in G.nodes():
        # 修正：正确访问每条边的权重
        total_weight = sum(data['weight'] for _, data in G[node].items())
        weighted_degrees[node] = total_weight
    return weighted_degrees
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
def calculate_k_shell(G):
    # 初始化节点的加权度数
    weighted_degrees = calculate_weighted_degree(G)
    max_degree = max(weighted_degrees.values())  # 获取最大加权度数
    node_to_shell = {}  # 记录每个节点的k-shell值

    # 初始化所有节点的k-shell为0
    for node in G.nodes():
        node_to_shell[node] = 0

    # 对每一个可能的k值，从1到最大加权度数迭代
    for k in range(1, int(max_degree) + 1):
        # 持续移除加权度数<=k的节点
        while True:
            to_remove = [node for node, deg in weighted_degrees.items() if deg <= k and node_to_shell[node] == 0]
            if not to_remove:
                break
            for node in to_remove:
                node_to_shell[node] = k  # 分配k-shell值
                # 更新相邻节点的加权度数
                for neighbor in list(G.neighbors(node)):
                    if neighbor in weighted_degrees:
                        weighted_degrees[neighbor] -= G[node][neighbor]['weight']
                # 该节点在后续迭代中不再考虑
                weighted_degrees[node] = float('inf')

    return node_to_shell

# 训练模型
# 主函数
def main():
    # 读取图并生成特征
    G = read_graph_from_txt()
    feature_matrix = generate_random_features(G, feature_dim=32)

    # 转换为 PyG 格式
    graph_data = from_networkx(G)
    graph_data.x = torch.tensor(feature_matrix, dtype=torch.float)

    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    model = GraphSAGE(feature_size=32, hidden_size=64, embedding_size=32).to(device)
    predictor = EdgePredictor(embedding_size=32).to(device)
    data = graph_data.to(device)

    # 训练参数
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=0.01)
    criterion = torch.nn.BCELoss()  # 二分类交叉熵损失

    # 训练模型
    for epoch in range(100):
        loss = train(model, predictor, data, optimizer, criterion, device)
        auc = evaluate(model, predictor, data, device)
        print(f'Epoch {epoch + 1}: Loss: {loss:.4f}, AUC: {auc:.4f}')

    # 预测缺失边的概率
    print("Starting edge prediction on GPU...")
    edge_probs = predict_edges(model, predictor,data)
    update_G(G, edge_probs)
    data = calculatenewlocation(G, "facebook_combinedinitial.txt")
    return data, G

if __name__ == "__main__":
    main()
