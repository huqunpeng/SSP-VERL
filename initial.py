import networkx as nx

def read_graph(file_path="celegans.txt"):
    return nx.read_edgelist(file_path, nodetype=int)

def read_node_values(file_path="celegansinitial.txt"):
    with open(file_path, "r") as f:
        return [float(line.strip()) for line in f]

def assign_signs_by_degree_and_neighbors(G, values):
    n = len(values)
    half_n = n // 2
    node_list = list(G.nodes())

    # 初始化全部为None（未赋值）
    signs = [None] * n

    # 根据度从大到小排序
    degree_sorted = sorted(node_list, key=lambda x: G.degree[x], reverse=True)

    negative_set = set()

    for node in degree_sorted:
        neighbors = list(G.neighbors(node))
        for nbr in neighbors:
            if signs[nbr] is None:
                signs[nbr] = -1
                negative_set.add(nbr)
                if len(negative_set) >= half_n:
                    break
        if len(negative_set) >= half_n:
            break

    # 剩下未赋值的为正
    for i in range(n):
        if signs[i] is None:
            signs[i] = 1

    # 构造最终有符号的节点值
    final_values = [abs(values[i]) * signs[i] for i in range(n)]

    return final_values, signs

def write_adjusted_values(values, output_file="celegansinitial.txt"):
    with open(output_file, "w") as f:
        for v in values:
            f.write(f"{v}\n")

# 主程序
if __name__ == "__main__":
    G = read_graph("celegans.txt")
    values = read_node_values("celegansinitial.txt")
    adjusted_values, signs = assign_signs_by_degree_and_neighbors(G, values)
    write_adjusted_values(adjusted_values)
    print("✅ 已根据度排序并设置符号，结果保存至 celegans_sign_assigned.txt")
