import networkx as nx
import random
import os

def extract_subgraphs(input_file, output_dir="./subgraphs",
                      num_subgraphs=10, depth_limit=2, max_nodes=100, seed=42, min_nodes=50):
    """
    从社交网络 txt 文件中提取若干个子图并保存为 txt 格式
    只保存节点数 ≥ min_nodes 的子图
    节点编号重映射为连续的 0..n-1
    最后一行不添加多余换行符
    """
    random.seed(seed)

    # 自动检测分隔符
    with open(input_file, "r") as f:
        first_line = f.readline()
        if "\t" in first_line:
            delimiter = "\t"
        elif "," in first_line:
            delimiter = ","
        else:
            delimiter = " "

    # 读取图
    G = nx.read_edgelist(input_file, delimiter=delimiter, nodetype=int)

    # 输出目录
    os.makedirs(output_dir, exist_ok=True)

    saved = 0
    trials = 0
    max_trials = num_subgraphs * 20  # 防止死循环

    while saved < num_subgraphs and trials < max_trials:
        trials += 1
        node = random.choice(list(G.nodes()))
        # BFS 扩展
        nodes = list(nx.bfs_tree(G, source=node, depth_limit=depth_limit).nodes())
        if len(nodes) > max_nodes:
            nodes = nodes[:max_nodes]
        subG = G.subgraph(nodes).copy()

        if subG.number_of_nodes() < min_nodes:
            continue  # 跳过小子图

        # 重新映射节点编号为 0..n-1
        subG = nx.convert_node_labels_to_integers(subG, ordering="default")

        out_path = os.path.join(output_dir, f"graph{saved+1}.txt")

        # 手动写，避免最后一行换行
        edges = list(subG.edges())
        with open(out_path, "w") as f:
            for j, (u, v) in enumerate(edges):
                if j < len(edges) - 1:
                    f.write(f"{u}{delimiter}{v}\n")
                else:
                    f.write(f"{u}{delimiter}{v}")  # 最后一行不加换行

        print(f"子图 {saved+1} 已保存到 {out_path} "
              f"(节点数={subG.number_of_nodes()}, 边数={subG.number_of_edges()})")
        saved += 1

    if saved < num_subgraphs:
        print(f"⚠️ 只找到 {saved} 个满足节点数 ≥ {min_nodes} 的子图，未达到要求的 {num_subgraphs} 个。")


# 使用示例
if __name__ == "__main__":
    extract_subgraphs(
        "./test_graphs/facebook_combined.txt",
        output_dir="./train_graphs",
        num_subgraphs=10,   # 需要 10 个子图
        depth_limit=2,
        max_nodes=100,
        seed=42,
        min_nodes=50        # 子图至少 50 节点
    )
