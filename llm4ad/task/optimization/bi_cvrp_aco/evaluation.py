from __future__ import annotations
from typing import Callable, List, Tuple
import numpy as np
import time
import random

from pymoo.indicators.hv import Hypervolume
from llm4ad.base import Evaluation
from llm4ad.task.optimization.bi_cvrp.get_instance import GetData
from llm4ad.task.optimization.bi_cvrp_aco.template import (
    template_program,
    task_description,
)

__all__ = ["BICVRPACOEvaluation"]

# --- Core Logic (Reused) ---
def compute_metrics(routes: list[np.ndarray], distance_matrix: np.ndarray):
    total_dist = 0.0
    max_dist = 0.0
    for route in routes:
        if len(route) <= 1: continue
        d = np.sum(distance_matrix[route[:-1], route[1:]])
        total_dist += d
        max_dist = max(max_dist, d)
    return total_dist, max_dist

def is_feasible(routes, demand, capacity):
    visited = set()
    for route in routes:
        if route[0] != 0 or route[-1] != 0: return False
        load = np.sum(demand[route])
        if load > capacity: return False
        for c in route[1:-1]:
            visited.add(c)
    n_customers = len(demand) - 1
    return len(visited) == n_customers and (not visited or max(visited) <= n_customers)

def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def random_initial_solution(num_nodes, demand, capacity):
    customers = list(range(1, num_nodes))
    random.shuffle(customers)
    routes = []
    current_route = [0]; current_load = 0.0
    for c in customers:
        if current_load + demand[c] <= capacity:
            current_route.append(c); current_load += demand[c]
        else:
            current_route.append(0)
            routes.append(np.array(current_route))
            current_route = [0, c]; current_load = demand[c]
    current_route.append(0)
    routes.append(np.array(current_route))
    return routes

# --- ACO Eval Loop ---
def evaluate_aco(instance_data, n_instance, ref_point, capacity_list, aco_func):
    hv_vals = []
    times = []
    all_objs_list = []

    for idx, (coords, demand, dist_mat) in enumerate(instance_data):
        cap = capacity_list[idx] if isinstance(capacity_list, list) else capacity_list
        start = time.time()

        # === IMPORTANT: RESET PHEROMONE ===
        if hasattr(aco_func, "pheromone"):
            del aco_func.pheromone
        # ==================================

        archive = []
        # Seed with random solutions
        for _ in range(10):
            sol = random_initial_solution(len(demand), demand, cap)
            obj = compute_metrics(sol, dist_mat)
            archive.append((sol, obj))
            
        # ACO Loop (1500 ants/iterations)
        for _ in range(1500):
            # Ant constructs solution
            s_new = aco_func(archive, coords, demand, dist_mat, cap)
            
            if not is_feasible(s_new, demand, cap):
                continue
                
            f_new = compute_metrics(s_new, dist_mat)
            
            # Update Pareto Archive
            if not any(dominates(f, f_new) for _, f in archive):
                archive = [(s, f) for s, f in archive if not dominates(f_new, f)]
                archive.append((s_new, f_new))
        
        elapsed = time.time() - start
        
        objs = np.array([f for _, f in archive])
        all_objs_list.append(objs)
        times.append(elapsed)

    all_concat = np.vstack(all_objs_list)
    z_ideal = np.min(all_concat, axis=0)
    z_nadir = np.max(all_concat, axis=0)
    
    hv = Hypervolume(ref_point=ref_point, norm_ref_point=False, zero_to_one=True, ideal=z_ideal, nadir=z_nadir)
    
    for objs in all_objs_list:
        hv_vals.append(-hv(objs))
        
    return np.mean(hv_vals), np.mean(times)

class BICVRPACOEvaluation(Evaluation):
    def __init__(self, **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=120 # ACO construction is slower than mutation
        )
        self.n_instance = 8
        self.problem_size = 50 
        getData = GetData(self.n_instance, self.problem_size)
        data = getData.generate_instances()
        if isinstance(data, tuple):
            self._datasets = data[0]
            self.cap = data[1]
        else:
            self._datasets = data
            self.cap = 200

        self.ref_point = np.array([1.1, 1.1])

    def evaluate_program(self, program_str: str, callable_func: Callable):
        return evaluate_aco(self._datasets, self.n_instance, self.ref_point, self.cap, callable_func)