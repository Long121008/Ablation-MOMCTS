from __future__ import annotations
from typing import Callable, List, Tuple
import numpy as np
import time
import random

from pymoo.indicators.hv import Hypervolume
from llm4ad.base import Evaluation
from llm4ad.task.optimization.tri_tsp_semo.get_instance import GetData
from llm4ad.task.optimization.tri_tsp_gls.template import (
    template_program,
    task_description,
)

__all__ = ["TSPGLSEvaluation"]


# --------------------------------------------------
# Core utilities
# --------------------------------------------------

def tour_cost(instance: np.ndarray, solution: np.ndarray) -> Tuple[float, float, float]:
    """Tri-objective tour length."""
    cost1, cost2, cost3 = 0.0, 0.0, 0.0
    n = len(solution)

    for i in range(n):
        a, b = int(solution[i]), int(solution[(i + 1) % n])
        
        # Extract coordinates for 3 objectives
        p1a, p2a, p3a = instance[a][:2], instance[a][2:4], instance[a][4:]
        p1b, p2b, p3b = instance[b][:2], instance[b][2:4], instance[b][4:]

        cost1 += np.linalg.norm(p1a - p1b)
        cost2 += np.linalg.norm(p2a - p2b)
        cost3 += np.linalg.norm(p3a - p3b)

    return cost1, cost2, cost3


def dominates(a, b):
    """Pareto dominance (minimization)."""
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
# GLS evaluation loop
# --------------------------------------------------

def evaluate_gls(
    instance_data,
    n_instance: int,
    problem_size: int,
    ref_point: np.ndarray,
    gls_operator: Callable,
):
    """
    Evaluate GLS-based TSP heuristic for Tri-objective.
    Returns: (mean_negative_HV, mean_runtime)
    """

    hv_values = []
    runtimes = []
    all_fronts = []

    # ---------- Per instance ----------
    for instance, dist1, dist2, dist3 in instance_data:
        start = time.time()
        
        # QUAN TRỌNG: Reset state penalties của GLS khi qua instance mới
        if hasattr(gls_operator, "penalties"):
            del gls_operator.penalties

        # Initial archive
        archive = []
        for _ in range(50):
            s = random_solution(problem_size)
            f = tour_cost(instance, s)
            archive.append((s, f))

        # Pareto filtering initial population
        archive = [
            (s, f)
            for s, f in archive
            if not any(dominates(f2, f) for _, f2 in archive if f2 != f)
        ]

        # ---------- GLS iterations ----------
        for _ in range(1500):
            # Pass 3 distance matrices for Tri-objective
            s_new = gls_operator(archive, instance, dist1, dist2, dist3)

            if not is_valid_tour(s_new, problem_size):
                continue

            f_new = tour_cost(instance, s_new)

            if not any(dominates(f, f_new) for _, f in archive):
                archive = [
                    (s, f)
                    for s, f in archive
                    if not dominates(f_new, f)
                ]
                archive.append((s_new, f_new))

        elapsed = time.time() - start

        objs = np.array([f for _, f in archive])
        all_fronts.append(objs)
        runtimes.append(elapsed)

    # ---------- Global HV normalization ----------
    if not all_fronts:
        return 0.0, 0.0
        
    all_concat = np.vstack(all_fronts)
    z_ideal = np.min(all_concat, axis=0)
    z_nadir = np.max(all_concat, axis=0)

    # Print bounds for debugging if needed
    # print(f"Z_ideal: {z_ideal}, Z_nadir: {z_nadir}")

    hv = Hypervolume(
        ref_point=ref_point,
        norm_ref_point=False, 
        zero_to_one=True,
        ideal=z_ideal,
        nadir=z_nadir,
    )

    for front in all_fronts:
        if len(front) == 0:
            hv_values.append(0.0)
        else:
            hv_values.append(-hv(front))  # negative for minimization

    return float(np.mean(hv_values)), float(np.mean(runtimes))


# --------------------------------------------------
# Evaluation class
# --------------------------------------------------

class TSPGLSEvaluation(Evaluation):
    """
    Evaluation for Tri-objective TSP using Guided Local Search (GLS)
    """

    def __init__(self, **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=90, # Increased timeout for 3 objectives
        )

        self.n_instance = 20
        self.problem_size = 20

        # Re-using the same generator as SEMO but will be used for GLS
        data_gen = GetData(self.n_instance, self.problem_size)
        self._datasets = data_gen.generate_instances()

        # HV reference for 3 objectives (normalized)
        self.ref_point = np.array([1.1, 1.1, 1.1])

    def evaluate_program(self, program_str: str, callable_func: Callable):
        return evaluate_gls(
            self._datasets,
            self.n_instance,
            self.problem_size,
            self.ref_point,
            callable_func,
        )