
template_program = '''
import numpy as np
from typing import List, Tuple
import random

# Global ACO State
pheromone = None
alpha = 1.0
beta = 2.0
rho = 0.1

def select_neighbor(
    archive: List[Tuple[List[np.ndarray], Tuple[float, float]]],
    coords: np.ndarray,
    demand: np.ndarray,
    distance_matrix: np.ndarray,
    capacity: float
) -> List[np.ndarray]:
    """
    Ant Colony Optimization (ACO) construction for Bi-CVRP.
    Constructs a full solution (set of routes) probabilistically.
    """
    global pheromone
    n_nodes = len(demand)
    
    # 1. Init Pheromone
    if pheromone is None:
        pheromone = np.ones((n_nodes, n_nodes))
        
    # 2. Heuristic (Inverse distance)
    heuristic = 1.0 / (distance_matrix + 1e-9)
    np.fill_diagonal(heuristic, 0)
    
    # 3. Construct Solution (One Ant)
    routes = []
    unvisited = set(range(1, n_nodes))
    
    while unvisited:
        current_node = 0 # Start at depot
        current_route = [0]
        current_load = 0.0
        
        while True:
            # Filter candidates that fit capacity
            candidates = [
                node for node in unvisited 
                if current_load + demand[node] <= capacity
            ]
            
            if not candidates:
                # Route full or no more customers fitting
                break
                
            # Calculate Probabilities
            probs = []
            for node in candidates:
                tau = pheromone[current_node, node]
                eta = heuristic[current_node, node]
                probs.append((tau ** alpha) * (eta ** beta))
            
            probs = np.array(probs)
            s_probs = probs.sum()
            
            if s_probs == 0:
                 # Fallback (very rare)
                 next_node = random.choice(candidates)
            else:
                 probs /= s_probs
                 next_node = random.choices(candidates, weights=probs, k=1)[0]
            
            # Move
            current_route.append(next_node)
            current_load += demand[next_node]
            unvisited.remove(next_node)
            current_node = next_node
            
        current_route.append(0) # Return to depot
        routes.append(np.array(current_route))
        
    # 4. Update Pheromone (Elitist & Evaporation)
    # Online update: Evaporate a bit and reinforce using Archive
    pheromone *= (1 - 0.05) # Small step evaporation
    
    # Reinforce edges from good solutions
    if archive:
        # Sample a few best solutions
        # Sort by scalarized cost (Dist + Makespan) usually, or random Pareto
        sample = random.sample(archive, min(len(archive), 3))
        
        for sol, (dist, span) in sample:
            cost = dist + span
            delta = 1.0 / (cost + 1e-9)
            
            for route in sol:
                for k in range(len(route) - 1):
                    u, v = route[k], route[k+1]
                    pheromone[u, v] += delta
                    pheromone[v, u] += delta

    return routes
'''

task_description = "Solve Bi-CVRP using Ant Colony Optimization (ACO). \
Construct a set of routes node-by-node. Start at depot (0). \
Probabilistically choose next customer based on Pheromone^alpha * Heuristic^beta, respecting capacity. \
If no customer fits, return to depot and start new route. \
Update global Pheromone matrix using solutions in Archive."




