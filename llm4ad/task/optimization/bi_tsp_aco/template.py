template_program = '''
import numpy as np
from typing import List, Tuple
import random

pheromone = None
alpha = 1.0
beta = 2.0
rho = 0.1

def select_neighbor(
    archive: List[Tuple[np.ndarray, Tuple[float, float]]],
    instance: np.ndarray,
    distance_matrix_1: np.ndarray,
    distance_matrix_2: np.ndarray
) -> np.ndarray:
    global pheromone

    n = len(archive[0][0])
    if pheromone is None:
        pheromone = np.ones((n, n))

    heuristic = 1.0 / (distance_matrix_1 + distance_matrix_2 + 1e-12)

    # ---- construct solution like an ant ----
    start = random.randint(0, n - 1)
    tour = [start]
    visited = set(tour)

    while len(tour) < n:
        i = tour[-1]
        probs = []
        candidates = []

        for j in range(n):
            if j not in visited:
                p = (pheromone[i, j] ** alpha) * (heuristic[i, j] ** beta)
                probs.append(p)
                candidates.append(j)

        probs = np.array(probs)
        probs /= probs.sum()

        next_city = random.choices(candidates, weights=probs)[0]
        tour.append(next_city)
        visited.add(next_city)

    # ---- pheromone update from archive (elitist) ----
    pheromone *= (1 - rho)
    for sol, obj in archive:
        cost = obj[0] + obj[1]
        for k in range(n):
            a = sol[k]
            b = sol[(k + 1) % n]
            pheromone[a, b] += 1.0 / cost
            pheromone[b, a] += 1.0 / cost

    return np.array(tour)
'''

task_description = "You are solving a Bi-objective Travelling Salesman Problem (bi-TSP), where each node has two different 2D coordinates: \
(x1, y1) and (x2, y2), representing its position in two objective spaces. The goal is to find a tour visiting each node exactly once and returning \
to the starting node, while minimizing two objectives simultaneously: the total tour length in each coordinate space. \
Given an archive of solutions, where each solution is a numpy array representing a TSP tour, and its corresponding objective \
is a tuple of two values (cost in each space), design a heuristic function named 'select_neighbor' that selects one solution from the archive \
and apply a novel or hybrid local search operator to generate a neighbor solution from it.  \
Must always ensure that the generated neighbor solution remains feasible, \
i.e., the solution must represent a valid TSP tour: it visits each node exactly once, ensuring no node is skipped or revisited.\
Please perform an intelligent random selection from among the solutions that show promising potential for further local improvement. Using a creative local search strategy that you design yourself, avoid 2-opt, \
go beyond standard approaches to design a method that yields higher-quality solutions across multiple objectives. The function should return the new neighbor solution."
