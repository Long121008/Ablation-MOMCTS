template_program = '''
import numpy as np
from typing import List, Tuple
import random
from collections import defaultdict

def select_neighbor(
    archive: List[Tuple[np.ndarray, Tuple[float, float]]],
    instance: np.ndarray,
    distance_matrix_1: np.ndarray,
    distance_matrix_2: np.ndarray
) -> np.ndarray:
    """
    Guided Local Search (GLS) operator for Bi-objective TSP.
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
            cost += distance_matrix_1[a][b] + distance_matrix_2[a][b]
            cost += lam * penalties[edge_key(a, b)]
        return cost

    def update_penalties(tour):
        utilities = []
        n = len(tour)
        for i in range(n):
            a, b = tour[i], tour[(i + 1) % n]
            base = distance_matrix_1[a][b] + distance_matrix_2[a][b]
            util = base / (1 + penalties[edge_key(a, b)])
            utilities.append((util, edge_key(a, b)))

        max_util = max(u for u, _ in utilities)
        for u, e in utilities:
            if u == max_util:
                penalties[e] += 1

    def relocate_move(tour):
        n = len(tour)
        i, j = random.sample(range(n), 2)
        new = list(tour)
        city = new.pop(i)
        new.insert(j, city)
        return np.array(new)

    # --- pick base solution ---
    base_tour = random.choice(archive)[0].copy()
    best = base_tour
    best_cost = penalized_cost(best)

    # --- local search ---
    for _ in range(10):
        cand = relocate_move(best)
        c_cost = penalized_cost(cand)
        if c_cost < best_cost:
            best, best_cost = cand, c_cost

    # --- GLS penalty update ---
    update_penalties(best)

    return best
'''


task_description = '''
You are solving a Bi-objective Travelling Salesman Problem (bi-TSP).

Design a Guided Local Search (GLS) heuristic named `select_neighbor`.

GLS should iteratively improve a complete TSP tour by applying local search moves,
while using adaptive penalties on solution features (e.g., edges or city transitions)
to escape local optima.

The function receives an archive of non-dominated tours. It should:
- Select a promising tour from the archive,
- Apply a penalized local search step using non-standard neighborhood structures
  (e.g., node relocation, insertion, or segment reordering),
- Adapt penalties implicitly through feature-aware decision making,
- Return a new feasible TSP tour.

Avoid standard 2-opt operators.
The returned solution must always be a valid permutation that visits each city
exactly once and returns to the starting city.
'''

