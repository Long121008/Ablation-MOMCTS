
template_program = '''
import numpy as np
from typing import List, Tuple
import random
from collections import defaultdict

def select_neighbor(
    archive: List[Tuple[np.ndarray, Tuple[float, float, float]]],
    instance: np.ndarray,
    distance_matrix_1: np.ndarray,
    distance_matrix_2: np.ndarray,
    distance_matrix_3: np.ndarray
) -> np.ndarray:
    """
    Guided Local Search (GLS) operator for Tri-objective TSP.
    Adapted to handle 3 distance matrices and persistent penalties.
    """

    # --- persistent penalty memory (per function instance) ---
    if not hasattr(select_neighbor, "penalties"):
        select_neighbor.penalties = defaultdict(int)
    penalties = select_neighbor.penalties

    def edge_key(a, b):
        return tuple(sorted((int(a), int(b))))

    def penalized_cost(tour, lam=0.1):
        cost = 0.0
        n = len(tour)
        for i in range(n):
            a, b = tour[i], tour[(i + 1) % n]
            # Sum of 3 objectives
            cost += distance_matrix_1[a][b] + distance_matrix_2[a][b] + distance_matrix_3[a][b]
            # Add penalty
            cost += lam * penalties[edge_key(a, b)]
        return cost

    def update_penalties(tour):
        utilities = []
        n = len(tour)
        for i in range(n):
            a, b = tour[i], tour[(i + 1) % n]
            # Base cost is sum of 3 distances
            base = distance_matrix_1[a][b] + distance_matrix_2[a][b] + distance_matrix_3[a][b]
            # Utility = Cost / (1 + Penalty)
            util = base / (1 + penalties[edge_key(a, b)])
            utilities.append((util, edge_key(a, b)))

        max_util = max(u for u, _ in utilities)
        for u, e in utilities:
            if u == max_util:
                penalties[e] += 1

    def relocate_move(tour):
        # A simple non-2-opt move: insert a city elsewhere
        n = len(tour)
        i, j = random.sample(range(n), 2)
        new = list(tour)
        city = new.pop(i)
        new.insert(j, city)
        return np.array(new)

    # --- pick base solution ---
    if not archive:
        return np.random.permutation(instance.shape[0])
        
    base_tour = random.choice(archive)[0].copy()
    best = base_tour
    best_cost = penalized_cost(best)

    # --- local search ---
    # Perform a few iterations of relocation to improve penalized cost
    for _ in range(10):
        cand = relocate_move(best)
        c_cost = penalized_cost(cand)
        if c_cost < best_cost:
            best, best_cost = cand, c_cost

    # --- GLS penalty update ---
    update_penalties(best)

    return best
'''

task_description = "You are solving a Tri-objective Travelling Salesman Problem (Tri-TSP). \
Design a Guided Local Search (GLS) heuristic named 'select_neighbor'. \
The function receives an archive of non-dominated tours and 3 distance matrices. \
It should: 1. Select a promising tour. 2. Apply a penalized local search (avoiding standard 2-opt, e.g., using relocation). \
3. Updates penalties for edges with high utility (high cost / low penalty) to escape local optima. \
Key: Combine all 3 objectives into the cost function and penalty logic."

