
template_program = '''
import numpy as np
from typing import List, Tuple
import random
from collections import defaultdict

def select_neighbor(
    archive: List[Tuple[np.ndarray, Tuple[float, float]]],
    weight_lst: np.ndarray,
    value1_lst: np.ndarray,
    value2_lst: np.ndarray,
    capacity: float 
) -> np.ndarray:
    """
    Guided Local Search (GLS) for Bi-objective Knapsack.
    Penalizes specific items to escape local optima.
    """
    
    # --- 1. Persistent Penalty Memory (per item) ---
    if not hasattr(select_neighbor, "penalties"):
        select_neighbor.penalties = defaultdict(float)
    penalties = select_neighbor.penalties
    
    n_items = len(weight_lst)
    lambda_param = 10.0 # Scaling factor for penalty

    # --- 2. Helper Functions ---
    def calculate_objectives(sol):
        # Return -1 if infeasible (overweight)
        w = np.sum(sol * weight_lst)
        if w > capacity: return -1.0, -1.0
        v1 = np.sum(sol * value1_lst)
        v2 = np.sum(sol * value2_lst)
        return v1, v2

    def augmented_score(sol):
        v1, v2 = calculate_objectives(sol)
        if v1 < 0: return -1e9 # Infeasible
        
        # Base score = Sum of objectives (scalarization)
        base = v1 + v2
        
        # Penalty: sum of penalties of selected items
        # We subtract penalty because we want to MAXIMIZE score
        p_val = 0.0
        indices = np.where(sol == 1)[0]
        for idx in indices:
            p_val += penalties[idx]
            
        return base - lambda_param * p_val

    def repair_and_optimize(sol):
        """Ensure feasibility and try to fill remaining space greedily."""
        current_w = np.sum(sol * weight_lst)
        indices = np.where(sol == 1)[0]
        
        # 1. Drop items if overweight (Drop lowest value/weight ratio first)
        if current_w > capacity:
            # Ratio based on sum of values
            ratios = (value1_lst[indices] + value2_lst[indices]) / weight_lst[indices]
            # Sort indices by ratio ascending
            sorted_idx_indices = np.argsort(ratios)
            
            for i in sorted_idx_indices:
                item_idx = indices[i]
                current_w -= weight_lst[item_idx]
                sol[item_idx] = 0
                if current_w <= capacity:
                    break
        
        # 2. Add items if space left (Add highest Augmented Ratio first)
        # Augmented Ratio = (v1 + v2 - lambda*penalty) / weight
        remaining_cap = capacity - np.sum(sol * weight_lst)
        if remaining_cap > 0:
            off_indices = np.where(sol == 0)[0]
            if len(off_indices) > 0:
                aug_vals = (value1_lst[off_indices] + value2_lst[off_indices] 
                            - lambda_param * np.array([penalties[i] for i in off_indices]))
                ratios = aug_vals / weight_lst[off_indices]
                
                # Sort descending
                sorted_off_args = np.argsort(ratios)[::-1]
                
                for i in sorted_off_args:
                    item_idx = off_indices[i]
                    if weight_lst[item_idx] <= remaining_cap:
                        sol[item_idx] = 1
                        remaining_cap -= weight_lst[item_idx]
                        
        return sol

    # --- 3. Main Logic ---
    # Pick a base solution
    if not archive:
        base_sol = np.zeros(n_items, dtype=int)
    else:
        base_sol, _ = random.choice(archive)
        base_sol = base_sol.copy()

    # Local Search: Mutate (Flip) then Repair
    # Try multiple mutations to find best neighbor based on Augmented Score
    best_neighbor = base_sol
    best_aug_score = augmented_score(base_sol)
    
    for _ in range(10): # 10 Local moves
        cand = base_sol.copy()
        # Flip 1 to 3 random bits
        n_flips = random.randint(1, 3)
        flip_indices = random.sample(range(n_items), n_flips)
        cand[flip_indices] = 1 - cand[flip_indices]
        
        cand = repair_and_optimize(cand)
        score = augmented_score(cand)
        
        if score > best_aug_score:
            best_neighbor = cand
            best_aug_score = score
            
    # --- 4. Update Penalties (GLS) ---
    # Penalize items in the local optimum that have high "Utility"
    # Utility = Contribution / (1 + Penalty)
    # Contribution ~ (v1 + v2) / weight
    
    selected_indices = np.where(best_neighbor == 1)[0]
    if len(selected_indices) > 0:
        contributions = (value1_lst[selected_indices] + value2_lst[selected_indices]) / weight_lst[selected_indices]
        current_penalties = np.array([penalties[i] for i in selected_indices])
        utilities = contributions / (1.0 + current_penalties)
        
        max_util = np.max(utilities)
        # Penalize all items close to max utility
        targets = np.where(utilities >= max_util - 1e-6)[0]
        
        for t in targets:
            item_idx = selected_indices[t]
            penalties[item_idx] += 1.0

    return best_neighbor
'''

task_description = "Solve Bi-objective Knapsack Problem (Bi-KP) using Guided Local Search (GLS). \
Maintain persistent 'penalties' for each item. \
Select a solution, apply local search (Flip + Greedy Repair) to maximize Augmented Score = (Val1+Val2) - lambda*Penalty. \
Update penalties for selected items with high utility (Value/Weight ratio normalized by penalty)."

