from __future__ import annotations
from typing import Callable, List, Tuple
import numpy as np
import time
import random

from pymoo.indicators.hv import Hypervolume
from llm4ad.base import Evaluation
from llm4ad.task.optimization.tri_tsp_semo.get_instance import GetData
from llm4ad.task.optimization.tri_tsp_aco.template import (
    template_program,
    task_description,
)

__all__ = ["TRITSPACOEvaluation"]


# --------------------------------------------------
# Core utilities
# --------------------------------------------------

def tour_cost(instance: np.ndarray, solution: np.ndarray) -> Tuple[float, float, float]:
    """Tri-objective tour length."""
    cost1, cost2, cost3 = 0.0, 0.0, 0.0
    n = len(solution)

    for i in range(n):
        a, b = int(solution[i]), int(solution[(i + 1) % n])
        
        p1a, p2a, p3a = instance[a][:2], instance[a][2:4], instance[a][4:]
        p1b, p2b, p3b = instance[b][:2], instance[b][2:4], instance[b][4:]

        cost1 += np.linalg.norm(p1a - p1b)
        cost2 += np.linalg.norm(p2a - p2b)
        cost3 += np.linalg.norm(p3a - p3b)

    return cost1, cost2, cost3


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def random_solution(n: int) -> np.ndarray:
    sol = list(range(n))
    random.shuffle(sol)
    return np.array(sol)


def is_valid_tour(sol: np.ndarray, n: int) -> bool:
    return (
        len(sol) == n and
        len(set(sol)) == n and
        np.all((sol >= 0) & (sol < n))
    )


# --------------------------------------------------
# ACO evaluation loop
# --------------------------------------------------

def evaluate_aco(
    instance_data,
    n_instance: int,
    problem_size: int,
    ref_point: np.ndarray,
    aco_construct_func: Callable,
):
    """
    Evaluate ACO-based TSP heuristic for Tri-objective.
    Returns: (mean_negative_HV, mean_runtime)
    """

    hv_vals = []
    times = []
    all_objs = []

    # ---------- run per instance ----------
    for instance, dist1, dist2, dist3 in instance_data:
        start = time.time()

        # QUAN TRỌNG: Reset pheromone khi qua map mới
        if hasattr(aco_construct_func, "pheromone"):
            del aco_construct_func.pheromone

        # Initial archive (Elite ants list)
        archive = []
        for _ in range(50):
            sol = random_solution(problem_size)
            obj = tour_cost(instance, sol)
            archive.append((sol, obj))

        # Pareto filtering
        archive = [
            (s, f)
            for s, f in archive
            if not any(dominates(f2, f) for _, f2 in archive if f2 != f)
        ]

        # ACO Loop: 1500 ants (iterations)
        # Note: In a real ACO, this might be grouped by generations (e.g., 30 gens x 50 ants)
        # But to match the function signature (one call = one ant/solution), we loop directly.
        for _ in range(1500):
            s_new = aco_construct_func(archive, instance, dist1, dist2, dist3)

            if not is_valid_tour(s_new, problem_size):
                continue

            f_new = tour_cost(instance, s_new)
            
            # Update Pareto Archive (Elitism)
            if not any(dominates(f, f_new) for _, f in archive):
                archive = [
                    (s, f)
                    for s, f in archive
                    if not dominates(f_new, f)
                ]
                archive.append((s_new, f_new))

        elapsed = time.time() - start

        objs = np.array([f for _, f in archive])
        all_objs.append(objs)
        times.append(elapsed)


    # ---------- Global HV normalization ----------
    if not all_objs:
        return 0.0, 0.0

    all_concat = np.vstack(all_objs)
    z_ideal = np.min(all_concat, axis=0)
    z_nadir = np.max(all_concat, axis=0)

    hv = Hypervolume(
        ref_point=ref_point,
        norm_ref_point=False, 
        zero_to_one=True,
        ideal=z_ideal,
        nadir=z_nadir,
    )

    for objs in all_objs:
        if len(objs) == 0:
             hv_vals.append(0.0)
        else:
            hv_vals.append(-hv(objs))  # negative for minimization

    return float(np.mean(hv_vals)), float(np.mean(times))


# --------------------------------------------------
# Evaluation class
# --------------------------------------------------

class TRITSPACOEvaluation(Evaluation):
    """
    Evaluation for Tri-objective TSP using Ant Colony Optimization (ACO)
    """

    def __init__(self, **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=90, # Increased for Tri-obj
        )

        self.n_instance = 20
        self.problem_size = 20

        data_gen = GetData(self.n_instance, self.problem_size)
        self._datasets = data_gen.generate_instances()

        # HV reference for 3 objectives
        self.ref_point = np.array([1.1, 1.1, 1.1])

    def evaluate_program(self, program_str: str, callable_func: Callable):
        return evaluate_aco(
            self._datasets,
            self.n_instance,
            self.problem_size,
            self.ref_point,
            callable_func,
        )