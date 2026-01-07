
template_program = '''
import numpy as np
from typing import List, Tuple
import random

# Global variables for ACO state
pheromone = None
alpha = 1.0
beta = 2.0
rho = 0.1

def select_neighbor(
    archive: List[Tuple[np.ndarray, Tuple[float, float, float]]],
    instance: np.ndarray,
    distance_matrix_1: np.ndarray,
    distance_matrix_2: np.ndarray,
    distance_matrix_3: np.ndarray
) -> np.ndarray:
    global pheromone

    n = len(instance) # Or len(archive[0][0])
    
    # Initialize pheromone matrix if first run
    if pheromone is None:
        pheromone = np.ones((n, n))

    # Heuristic: Inverse of sum of all 3 distances
    # Adding epsilon to avoid division by zero
    total_distance = distance_matrix_1 + distance_matrix_2 + distance_matrix_3
    heuristic = 1.0 / (total_distance + 1e-12)

    # ---- construct solution like an ant ----
    start = random.randint(0, n - 1)
    tour = [start]
    visited = set(tour)

    while len(tour) < n:
        i = tour[-1]
        probs = []
        candidates = []

        # Identify valid candidates
        possible_next = [node for node in range(n) if node not in visited]
        
        if not possible_next:
            break

        for j in possible_next:
            # ACO Probability Formula: (tau^alpha) * (eta^beta)
            p = (pheromone[i, j] ** alpha) * (heuristic[i, j] ** beta)
            probs.append(p)
            candidates.append(j)

        probs = np.array(probs)
        s = probs.sum()
        if s == 0:
            # Fallback if probs are zero (rare)
            probs = np.ones(len(probs)) / len(probs)
        else:
            probs /= s

        # Roulette Wheel Selection
        next_city = random.choices(candidates, weights=probs, k=1)[0]
        tour.append(next_city)
        visited.add(next_city)

    tour = np.array(tour)

    # ---- pheromone update from archive (elitist) ----
    # 1. Evaporation
    pheromone *= (1 - rho)
    
    # 2. Reinforcement using solutions in Archive
    for sol, obj in archive:
        # Cost is sum of 3 objective values
        cost = obj[0] + obj[1] + obj[2]
        if cost == 0: cost = 1e-9
        
        delta_tau = 1.0 / cost
        
        for k in range(n):
            a = sol[k]
            b = sol[(k + 1) % n]
            pheromone[a, b] += delta_tau
            pheromone[b, a] += delta_tau # Symmetric TSP

    return tour
'''

task_description = "You are solving a Tri-objective Travelling Salesman Problem (Tri-TSP). \
Design an Ant Colony Optimization (ACO) heuristic named 'select_neighbor'. \
Use global variables to maintain 'pheromone' state. \
Logic: 1. Calculate heuristic as 1.0 / (sum of 3 distance matrices). \
2. Construct a solution probabilistically (ant walk). \
3. Apply evaporation and update pheromones using the high-quality solutions found in the archive (Elitist update). \
Return the constructed tour."