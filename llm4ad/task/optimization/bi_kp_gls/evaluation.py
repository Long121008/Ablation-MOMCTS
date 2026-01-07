from __future__ import annotations
from typing import Callable, List, Tuple
import numpy as np
import time
import random

from pymoo.indicators.hv import Hypervolume
from llm4ad.base import Evaluation
from llm4ad.task.optimization.bi_kp.get_instance import GetData
from llm4ad.task.optimization.bi_kp_gls.template import (
    template_program,
    task_description,
)

__all__ = ["BIKPGLSEvaluation"]

# --- Core Utilities ---
def knapsack_value(solution: np.ndarray, weight_lst: np.ndarray, value1_lst: np.ndarray, value2_lst: np.ndarray, capacity: float):
    w = np.sum(solution * weight_lst)
    if w > capacity:
        return -1e10, -1e10  # Penalize infeasible
    if not np.all(np.isin(solution, [0, 1])):
        return -1e10, -1e10
    total_val1 = np.sum(solution * value1_lst)
    total_val2 = np.sum(solution * value2_lst)
    return total_val1, total_val2

def dominates(a, b):
    # a dominates b if a >= b and a > b in at least one obj
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))

def random_solution(weight_lst, capacity, problem_size):
    sol = np.zeros(problem_size, dtype=int)
    idxs = list(range(problem_size))
    random.shuffle(idxs)
    current_w = 0.0
    for i in idxs:
        if current_w + weight_lst[i] <= capacity:
            sol[i] = 1
            current_w += weight_lst[i]
    return sol

# --- GLS Eval Loop ---
def evaluate_gls(instance_data, n_instance, ref_point, problem_size, capacity_data, gls_func):
    hv_vals = []
    times = []
    
    for idx, (weight_lst, value1_lst, value2_lst) in enumerate(instance_data):
        # Handle capacity being a list or scalar depending on GetData impl
        cap = capacity_data[idx] if isinstance(capacity_data, list) else capacity_data
        start = time.time()

        # === IMPORTANT: RESET GLS STATE ===
        if hasattr(gls_func, "penalties"):
            del gls_func.penalties
        # ==================================

        # Init Archive
        archive = []
        for _ in range(20):
            s = random_solution(weight_lst, cap, problem_size)
            obj = knapsack_value(s, weight_lst, value1_lst, value2_lst, cap)
            if obj[0] > -1e5:
                archive.append((s, obj))
        
        # Pareto Filter
        archive = [
            (s, f) for s, f in archive
            if not any(dominates(f2, f) for _, f2 in archive if f2 != f)
        ]

        # GLS Loop (4000 iters as per original eval)
        for _ in range(4000):
            s_prime = gls_func(archive, weight_lst, value1_lst, value2_lst, cap)
            f_prime = knapsack_value(s_prime, weight_lst, value1_lst, value2_lst, cap)
            
            if f_prime[0] < -1e5: continue

            if not any(dominates(f_a, f_prime) for _, f_a in archive):
                archive = [(a, f_a) for a, f_a in archive if not dominates(f_prime, f_a)]
                archive.append((s_prime, f_prime))

        elapsed = time.time() - start
        
        objs = np.array([f for _, f in archive])
        
        # Compute HV for this instance
        if len(objs) == 0:
            hv_vals.append(0.0)
        else:
            # Need local ideal/nadir for correct normalization or use global?
            # Using per-instance normalization as typically done in simple evals
            z_ideal = np.min(objs, axis=0)
            z_nadir = np.max(objs, axis=0) + 1e-6
            
            hv = Hypervolume(ref_point=ref_point, norm_ref_point=False, zero_to_one=True, ideal=z_ideal, nadir=z_nadir)
            hv_vals.append(-hv(objs)) # Negative for minimization
            
        times.append(elapsed)

    return np.mean(hv_vals), np.mean(times)

class BIKPGLSEvaluation(Evaluation):
    def __init__(self, **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=90
        )
        self.n_instance = 8
        self.problem_size = 200
        getData = GetData(self.n_instance, self.problem_size)
        data = getData.generate_instances()
        # Handle unpacking depending on GetData return signature
        if isinstance(data, tuple):
            self._datasets = data[0]
            self.cap = data[1]
        else:
            self._datasets = data
            self.cap = 500 # fallback
            
        self.ref_point = np.array([1.1, 1.1])

    def evaluate_program(self, program_str: str, callable_func: Callable):
        return evaluate_gls(self._datasets, self.n_instance, self.ref_point, self.problem_size, self.cap, callable_func)