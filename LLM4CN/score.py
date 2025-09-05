import random
import multiprocessing
import time
from utils import sample_by_degree_distribution, calculate_anc, calculate_anc_gcc, sorted_by_value
from scipy.sparse import load_npz

def run_algorithm(edge_matrix, number_of_removed_nodes, metric, algorithm_code, result_queue, return_nodes=False):
    try:
        globals_dict = {}
        exec(algorithm_code, globals_dict)
        result_dict = globals_dict['score_nodes'](edge_matrix)
        result = sorted_by_value(result_dict)
        top_nodes = result[:number_of_removed_nodes]

        if metric == 'pc':
            score = 1 - calculate_anc(edge_matrix, top_nodes)
        elif metric == 'gcc':
            score = 1 - calculate_anc_gcc(edge_matrix, top_nodes)
        else:
            score = 0

        if return_nodes:
            result_queue.put((score, top_nodes))
        else:
            result_queue.put(score)

    except Exception as e:
        print("This code can not be evaluated!")
        if return_nodes:
            result_queue.put((0, []))
        else:
            result_queue.put(0)

class ScoringModule:

    def __init__(self, file_path, ratio, calculate_type='pc'):
        self.edge_matrix = load_npz(file_path).toarray()
        if len(self.edge_matrix) > 5000:
            self.edge_matrix = sample_by_degree_distribution(self.edge_matrix, 0.1)
        self.number_of_removed_nodes = int(ratio * 0.01 * len(self.edge_matrix))
        self.metric = calculate_type
        print(f'removed number: {self.number_of_removed_nodes}, metric: {self.metric}')

    def score_nodes_with_timeout(self, algorithm, timeout=60):
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=run_algorithm,
            args=(
                self.edge_matrix,
                self.number_of_removed_nodes,
                self.metric,
                algorithm,
                result_queue,
                False,  # ← 不返回节点列表
            )
        )

        process.start()
        process.join(timeout)

        if process.is_alive():
            print("Algorithm execution exceeded timeout. Terminating...")
            process.terminate()
            process.join()
            return 0
        else:
            return result_queue.get()

    def score_nodes_with_result(self, algorithm, timeout=60):
        """返回评分和种子节点列表"""
        result_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=run_algorithm,
            args=(
                self.edge_matrix,
                self.number_of_removed_nodes,
                self.metric,
                algorithm,
                result_queue,
                True,  # ← 返回节点列表
            )
        )

        process.start()
        process.join(timeout)

        if process.is_alive():
            print("Algorithm execution exceeded timeout. Terminating...")
            process.terminate()
            process.join()
            return 0, []
        else:
            return result_queue.get()

    def evaluate_algorithm(self, algorithm):
        return self.score_nodes_with_timeout(algorithm, timeout=60)

if __name__ == '__main__':
    print('Score Module')
