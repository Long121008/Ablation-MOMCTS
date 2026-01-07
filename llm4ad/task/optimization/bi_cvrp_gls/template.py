
template_program = '''
import numpy as np
from typing import List, Tuple
import random
from collections import defaultdict

def select_neighbor(
    archive: List[Tuple[List[np.ndarray], Tuple[float, float]]],
    coords: np.ndarray,
    demand: np.ndarray,
    distance_matrix: np.ndarray,
    capacity: float
) -> List[np.ndarray]:
    """
    Guided Local Search (GLS) for Bi-CVRP.
    Uses persistent penalties on edges to guide local search out of local optima.
    """
    
    # --- 1. Persistent Penalty Memory ---
    if not hasattr(select_neighbor, "penalties"):
        select_neighbor.penalties = defaultdict(int)
    penalties = select_neighbor.penalties

    # Helper: Unique key for an undirected edge (u, v)
    def edge_key(u, v):
        return tuple(sorted((int(u), int(v))))

    # --- 2. Cost Functions ---
    def calculate_obj(sol):
        total_dist = 0.0
        max_dist = 0.0
        for route in sol:
            d = 0.0
            for i in range(len(route) - 1):
                d += distance_matrix[route[i], route[i+1]]
            total_dist += d
            if d > max_dist: max_dist = d
        return total_dist, max_dist

    def augmented_cost(sol, lam=0.5):
        # Cost = (Dist + Makespan) + lambda * Penalty
        dist, span = calculate_obj(sol)
        base_cost = dist + span
        
        penalty_val = 0.0
        for route in sol:
            for i in range(len(route) - 1):
                penalty_val += penalties[edge_key(route[i], route[i+1])]
        
        return base_cost + lam * penalty_val

    # --- 3. Local Search Operators ---
    def get_random_route_indices(sol):
        if not sol: return None, None
        r_idx = random.randint(0, len(sol) - 1)
        # return route index and a random customer index within that route (excluding depot 0 at ends)
        if len(sol[r_idx]) <= 2: return r_idx, -1 # Empty route or just depots
        c_idx = random.randint(1, len(sol[r_idx]) - 2)
        return r_idx, c_idx

    def relocate(sol):
        # Move a customer from one route to another (or same route)
        new_sol = [r.copy() for r in sol]
        
        r1_idx, c1_idx = get_random_route_indices(new_sol)
        if c1_idx == -1: return sol # Cannot move
        
        customer = new_sol[r1_idx][c1_idx]
        
        # Remove
        new_sol[r1_idx] = np.delete(new_sol[r1_idx], c1_idx)
        
        # Insert elsewhere
        r2_idx = random.randint(0, len(new_sol) - 1)
        insert_idx = random.randint(1, len(new_sol[r2_idx]) - 1)
        
        new_sol[r2_idx] = np.insert(new_sol[r2_idx], insert_idx, customer)
        
        # Check capacity
        if sum(demand[new_sol[r2_idx]]) > capacity:
            return sol # Invalid move
            
        # Clean up empty routes (preserving at least one if needed, but here usually we remove [0,0])
        new_sol = [r for r in new_sol if len(r) > 2 or sum(demand[r]) > 0]
        if not new_sol: new_sol = [np.array([0, 0])]
        
        return new_sol

    # --- 4. Main Logic ---
    # Pick a base solution
    if not archive:
        # Fallback (should not happen usually)
        return [np.array([0, 0])]
        
    base_sol, _ = random.choice(archive)
    current_sol = [r.copy() for r in base_sol]
    current_aug_cost = augmented_cost(current_sol)
    
    # Apply Local Search (Greedy Descent on Augmented Cost)
    # Try 20 modifications
    for _ in range(20):
        neighbor = relocate(current_sol)
        # Simplified: if valid and better augmented cost, accept
        # Note: relocate already checks capacity
        neigh_aug_cost = augmented_cost(neighbor)
        if neigh_aug_cost < current_aug_cost:
            current_sol = neighbor
            current_aug_cost = neigh_aug_cost

    # --- 5. Update Penalties (GLS) ---
    # Find edges with max utility in the local optimum
    max_util = -1.0
    edges_to_penalize = []
    
    for route in current_sol:
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            dist = distance_matrix[u, v]
            p = penalties[edge_key(u, v)]
            util = dist / (1.0 + p)
            
            if util > max_util:
                max_util = util
                edges_to_penalize = [(u, v)]
            elif abs(util - max_util) < 1e-6:
                edges_to_penalize.append((u, v))
                
    for u, v in edges_to_penalize:
        penalties[edge_key(u, v)] += 1
        
    return current_sol
'''

task_description = "Solve Bi-CVRP using Guided Local Search (GLS). \
Maintain persistent 'penalties' on edges. \
Select a solution, apply local search (Relocate/Swap) to minimize Augmented Cost = (Dist+Makespan) + lambda*Penalty. \
Update penalties for high-utility edges in the resulting solution."




