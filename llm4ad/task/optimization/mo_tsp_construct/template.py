template_program = '''
import numpy as np
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int: 
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    next_node = unvisited_nodes[0]

    return next_node
'''

task_description = "You are given a set of nodes with their coordinates, and the objective is to construct a route that visits each node exactly once and returns to the starting node. The primary goal is to minimize the total travel distance of the route, while the secondary goal is to ensure that the algorithm achieves this with efficient runtime. The problem should be solved in a step-by-step manner by starting from the current node and iteratively selecting the next node to visit. Your task is to design a novel node-selection strategy that balances accuracy (shorter total distance) with computational efficiency (faster selection process), rather than relying on existing algorithms in the literature."