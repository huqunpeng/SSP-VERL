import torch
import networkx as nx
import Utils
import time
def glstm(G, k=40):
    model = torch.load('GLSTM.pth')  # 载入迁移学习模型
    embedding = Utils.embedding_(G)  # 计算节点嵌入
    model.eval()

    value = model(embedding)  # 预测影响力
    nodes_list = list(G.nodes())
    prediction_I = value.detach().numpy()

    # 绑定节点和影响力分数
    prediction_I_with_node = [[nodes_list[i], prediction_I[i][0]] for i in range(len(prediction_I))]
    prediction_I_with_node.sort(key=lambda x: x[1], reverse=True)  # 按影响力得分排序

    # 选取前 k 个节点作为种子节点
    seed_nodes = [x[0] for x in prediction_I_with_node[:k]]

    return seed_nodes
def drawgraph():
    G = nx.read_edgelist("pester.txt", nodetype=int)
    return G
start=time.time()
G=drawgraph()
for i in range(100):
    seed_nodes = glstm(G, k=40)  # 选取 20 个影响力最大的种子节点
print("选出的种子节点:", seed_nodes)
end=time.time()
print(end-start)
