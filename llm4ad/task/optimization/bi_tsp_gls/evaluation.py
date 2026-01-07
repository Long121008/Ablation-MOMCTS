from __future__ import annotations
from typing import Callable, List, Tuple
import numpy as np
import time
import random

from pymoo.indicators.hv import Hypervolume
from llm4ad.base import Evaluation
from llm4ad.task.optimization.bi_tsp_gls.get_instance import GetData
from llm4ad.task.optimization.bi_tsp_gls.template import (
    template_program,
    task_description,
)

__all__ = ["TSPGLSEvaluation"]


# --------------------------------------------------
# Core utilities
# --------------------------------------------------

def tour_cost(instance: np.ndarray, solution: np.ndarray) -> Tuple[float, float]:
    """Bi-objective tour length."""
    c1, c2 = 0.0, 0.0
    n = len(solution)

    for i in range(n):
        a, b = solution[i], solution[(i + 1) % n]
        p1a, p2a = instance[a][:2], instance[a][2:4]
        p1b, p2b = instance[b][:2], instance[b][2:4]
        c1 += np.linalg.norm(p1a - p1b)
        c2 += np.linalg.norm(p2a - p2b)

    return c1, c2


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
    Evaluate GLS-based TSP heuristic.
    Returns: (mean_negative_HV, mean_runtime)
    """

    hv_values = []
    runtimes = []
    all_fronts = []

    # ---------- Per instance ----------
    for instance, dist1, dist2 in instance_data:
        start = time.time()

        # Initial archive
        archive = []
        for _ in range(50):
            s = random_solution(problem_size)
            f = tour_cost(instance, s)
            archive.append((s, f))

        # Pareto filtering
        archive = [
            (s, f)
            for s, f in archive
            if not any(dominates(f2, f) for _, f2 in archive if f2 != f)
        ]

        # ---------- GLS iterations ----------
        for _ in range(1500):
            s_new = gls_operator(archive, instance, dist1, dist2)

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
    all_concat = np.vstack(all_fronts)
    z_ideal = np.min(all_concat, axis=0)
    z_nadir = np.max(all_concat, axis=0)

    hv = Hypervolume(
        ref_point=ref_point,
        norm_ref_point=False, 
        zero_to_one=True,
        ideal=z_ideal,
        nadir=z_nadir,
    )

    for front in all_fronts:
        hv_values.append(-hv(front))  # negative for minimization

    return float(np.mean(hv_values)), float(np.mean(runtimes))


# --------------------------------------------------
# Evaluation class
# --------------------------------------------------

class TSPGLSEvaluation(Evaluation):
    """
    Evaluation for Bi-objective TSP using Guided Local Search (GLS)
    """

    def __init__(self, **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=60,
        )

        self.n_instance = 20
        self.problem_size = 20

        data_gen = GetData(self.n_instance, self.problem_size)
        self._datasets = data_gen.generate_instances()

        # HV reference (normalized)
        self.ref_point = np.array([1.1, 1.1])

    def evaluate_program(self, program_str: str, callable_func: Callable):
        return evaluate_gls(
            self._datasets,
            self.n_instance,
            self.problem_size,
            self.ref_point,
            callable_func,
        )
