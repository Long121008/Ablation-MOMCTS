from __future__ import annotations
from typing import Callable, List, Tuple
import numpy as np
import time
import random

from pymoo.indicators.hv import Hypervolume
from llm4ad.base import Evaluation
from llm4ad.task.optimization.bi_tsp_aco.get_instance import GetData
from llm4ad.task.optimization.bi_tsp_aco.template import (
    template_program,
    task_description,
)

__all__ = ["TSPACOEvaluation"]


def tour_cost(instance: np.ndarray, solution: np.ndarray) -> Tuple[float, float]:
    cost1, cost2 = 0.0, 0.0
    n = len(solution)

    for i in range(n):
        a, b = solution[i], solution[(i + 1) % n]
        p1a, p2a = instance[a][:2], instance[a][2:4]
        p1b, p2b = instance[b][:2], instance[b][2:4]

        cost1 += np.linalg.norm(p1a - p1b)
        cost2 += np.linalg.norm(p2a - p2b)

    return cost1, cost2


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def random_solution(n: int) -> np.ndarray:
    sol = list(range(n))
    random.shuffle(sol)
    return np.array(sol)


def evaluate_aco(
    instance_data,
    n_instance: int,
    problem_size: int,
    ref_point: np.ndarray,
    neighbor_func: Callable,
):
    hv_vals = []
    times = []

    all_objs = []

    # ---------- run per instance ----------
    for instance, dist1, dist2 in instance_data:
        start = time.time()

        # Initial archive
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

        for _ in range(1500):
            s_new = neighbor_func(archive, instance, dist1, dist2)

            if len(set(s_new)) != problem_size:
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
        all_objs.append(objs)
        times.append(elapsed)


    all_concat = np.vstack(all_objs)
    z_ideal = np.min(all_concat, axis=0)
    z_nadir = np.max(all_concat, axis=0)

    print(f"Check bound: {z_ideal, z_nadir}")
    hv = Hypervolume(
        ref_point=ref_point,
        norm_ref_point=False, 
        zero_to_one=True,
        ideal=z_ideal,
        nadir=z_nadir,
    )

    for objs in all_objs:
        hv_vals.append(-hv(objs))  # negative for minimization

    print(f"Check hv_array: {hv_vals}")
    return float(np.mean(hv_vals)), float(np.mean(times))

class TSPACOEvaluation(Evaluation):
    """
    Evaluation for Bi-objective TSP using Ant Colony Optimization (ACO)
    """

    def __init__(self, **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=500,
        )

        self.n_instance = 10
        self.problem_size = 150

        data_gen = GetData(self.n_instance, self.problem_size)
        self._datasets = data_gen.generate_instances()

        # HV reference (normalized)
        self.ref_point = np.array([1.1, 1.1])

    def evaluate_program(self, program_str: str, callable_func: Callable):
        return evaluate_aco(
            self._datasets,
            self.n_instance,
            self.problem_size,
            self.ref_point,
            callable_func,
        )
