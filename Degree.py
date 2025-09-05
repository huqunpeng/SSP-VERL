import networkx as nx
import time
def read_graph_from_txt():
    G = nx.read_edgelist("google12.txt", nodetype=int)
    for u, v in G.edges():
        G[u][v]["weight"] = 1
    return G
def degree_algorithm(G, num_nodes):
    degree_dict = dict(G.degree())
    sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
    selected_nodes = sorted_nodes[:num_nodes]
    return selected_nodes


def main():
    file_path = 'google12.txt'  # Replace with the path to your txt file
    num_nodes_to_select = 30  # Replace with the desired number of nodes to select

    G = read_graph_from_txt()
    for i in range(100):
        selected_nodes = degree_algorithm(G, num_nodes_to_select)

    print("Selected nodes for influence maximization:", selected_nodes)


if __name__ == "__main__":
    start=time.time()
    main()
    end=time.time()
    print(end-start)
