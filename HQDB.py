import networkx as nx
import time

def read_graph_from_txt(file_path):
    G = nx.Graph()
    with open(file_path, 'r') as file:
        for line in file:
            node1, node2 = line.strip().split()
            G.add_edge(node1, node2)
    return G

def hqdb_algorithm(G, num_nodes):
    degree_dict = dict(G.degree())
    quality_scores = degree_dict.copy()
    selected_nodes = []

    for _ in range(num_nodes):
        max_node = max(quality_scores, key=quality_scores.get)
        selected_nodes.append(max_node)

        for neighbor in G.neighbors(max_node):
            if neighbor not in selected_nodes:
                quality_scores[neighbor] -= 1

        quality_scores[max_node] = -1

    return selected_nodes

def main():
    file_path = 'google12.txt'
    G = read_graph_from_txt(file_path)

    results = {}

    for num_nodes_to_select in range(5, 55, 5):
        start_time = time.time()

        for _ in range(100):
            selected_nodes = hqdb_algorithm(G, num_nodes_to_select)

        end_time = time.time()
        elapsed_time = end_time - start_time

        results[num_nodes_to_select] = elapsed_time
        print(f"选择种子节点数={num_nodes_to_select}, 耗时={elapsed_time:.4f}s")

    print("\n所有运行时间：")
    for k, t in results.items():
        print(t)

if __name__ == "__main__":
    main()
