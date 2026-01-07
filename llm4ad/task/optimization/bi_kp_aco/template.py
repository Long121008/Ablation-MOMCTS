
template_program = '''
import numpy as np
from typing import List, Tuple
import random

# Global ACO variables
pheromone = None
alpha = 1.0
beta = 2.0
rho = 0.05

def select_neighbor(
    archive: List[Tuple[np.ndarray, Tuple[float, float]]],
    weight_lst: np.ndarray,
    value1_lst: np.ndarray,
    value2_lst: np.ndarray,
    capacity: float 
) -> np.ndarray:
    """
    Ant Colony Optimization (ACO) construction for Bi-KP.
    Constructs a solution item-by-item based on Pheromone and Heuristic.
    """
    global pheromone
    n_items = len(weight_lst)
    
    # 1. Initialize Pheromone (Vector of size N_items)
    if pheromone is None:
        pheromone = np.ones(n_items) * 0.5
        
    # 2. Heuristic Info: Average Profit / Weight
    # Combine both objectives
    avg_value = (value1_lst + value2_lst)
    heuristic = avg_value / (weight_lst + 1e-9)
    
    # 3. Construct Solution (One Ant)
    solution = np.zeros(n_items, dtype=int)
    current_weight = 0.0
    candidates = list(range(n_items))
    
    while candidates:
        # Filter candidates that fit
        feasible_candidates = [
            i for i in candidates 
            if current_weight + weight_lst[i] <= capacity
        ]
        
        if not feasible_candidates:
            break
            
        # Calculate probabilities
        probs = []
        for i in feasible_candidates:
            tau = pheromone[i]
            eta = heuristic[i]
            p = (tau ** alpha) * (eta ** beta)
            probs.append(p)
            
        probs = np.array(probs)
        s_probs = probs.sum()
        
        if s_probs == 0:
            # Random selection if probs zero
            idx = random.choice(range(len(feasible_candidates)))
            selected_item = feasible_candidates[idx]
        else:
            # Roulette Wheel
            probs /= s_probs
            idx = np.random.choice(range(len(feasible_candidates)), p=probs)
            selected_item = feasible_candidates[idx]
            
        # Add item
        solution[selected_item] = 1
        current_weight += weight_lst[selected_item]
        candidates.remove(selected_item)
        
    # 4. Update Pheromone (Evaporation & Reinforcement)
    # Evaporate global pheromone
    pheromone *= (1 - rho)
    
    # Reinforce using Archive (Elitism)
    # Items present in high-quality solutions get pheromone boost
    if archive:
        # Reinforce all Pareto solutions or a subset
        for sol, vals in archive:
            # Reinforcement amount could be proportional to quality, 
            # here we use constant or simple weight
            delta = 0.1 
            # Items selected in this solution get boost
            indices = np.where(sol == 1)[0]
            pheromone[indices] += delta
            
    return solution
'''
task_description = "Solve Bi-objective Knapsack Problem (Bi-KP) using Ant Colony Optimization (ACO). \
Maintain global 'pheromone' vector for items. \
Construct solution by probabilistically selecting items (Pheromone^alpha * Heuristic^beta) until capacity full. \
Update pheromone by evaporating and reinforcing items present in the Archive solutions."