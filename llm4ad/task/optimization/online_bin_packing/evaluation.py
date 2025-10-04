# Module Name: OBPEvaluation
# Last Revision: 2025/2/16
# Description: Evaluates the Online Bin Packing Problem (OBP).
#              Given a sequence of items arriving one by one, the goal is to pack them into bins
#              of fixed capacity in real-time, minimizing the number of bins used.
#              This module is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Parameters:
#    - timeout_seconds: Maximum allowed time (in seconds) for the evaluation process: int (default: 30).
#    - n_instances: Number of problem instances to generate: int (default: 5).
#    - n_items: Number of items to pack: int (default: 5000).
#    - capacity: Maximum capacity of each bin: int (default: 100).
#
# References:
#   - Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
#       Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
#       with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
# 
# Permission is granted to use the LLM4AD platform for research purposes. 
# All publications, software, or other works that utilize this platform 
# or any part of its codebase must acknowledge the use of "LLM4AD" and 
# cite the following reference:
# 
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
# 
# For inquiries regarding commercial use or licensing, please contact 
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import time

from llm4ad.base import Evaluation
from llm4ad.task.optimization.online_bin_packing.template import template_program, task_description
from llm4ad.task.optimization.online_bin_packing.generate_weibull_instances import generate_weibull_dataset

__all__ = ['OBPEvaluation']


class OBPEvaluation(Evaluation):
    """Evaluator for online bin packing problem."""

    def __init__(self, timeout_seconds=30,
                 n_instances=5,
                 n_items=5000,
                 capacity=100,
                 **kwargs):
        """
        Args:
            - 'data_file' (str): The data file to load (default is 'weibull_5k_train.pkl').
            - 'data_key' (str): The key of the data to load (default is 'data_key').

        Raises:
            AttributeError: If the data key does not exist.
            FileNotFoundError: If the specified data file is not found.
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        self.n_instances = n_instances
        self.n_items = n_items
        self.capacity = capacity

        self._datasets = generate_weibull_dataset(self.n_instances, self.n_items, self.capacity)

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        # 🌟 Gọi hàm evaluate và lấy cả hiệu suất và thời gian
        results, avg_time = self.evaluate(callable_func)
        return results, avg_time 


    def plot_solution(self, bins_packed: np.ndarray, items: list, capacity: int, max_unused_bins: int = 5):
        """
        Plot the solution of the 1D Online Bin Packing Problem, omitting unused bins.

        Args:
            bins_packed: A numpy array of remaining capacities in the bins after packing.
            items: A list of item sizes.
            capacity: The capacity of each bin.
            max_unused_bins: Maximum number of unused bins to include in the plot (for sampling).
        """
        # Calculate the number of bins used
        num_bins = (bins_packed != capacity).sum()

        #
        n_show = 15

        # Check for empty bins or invalid inputs
        if num_bins == 0:
            print("No bins used.")
            return
        if len(items) == 0:
            print("No items to pack.")
            return

        # Track which items are assigned to which bins
        item_assignment = [[] for _ in range(len(bins_packed))]
        current_bin = 0
        current_position = 0

        for item in items:
            if current_bin >= len(bins_packed):
                break  # No more bins available
            if current_position + item <= capacity - bins_packed[current_bin]:
                item_assignment[current_bin].append((current_position, item))
                current_position += item
            else:
                current_bin += 1
                current_position = 0
                if current_bin >= len(bins_packed):
                    break
                item_assignment[current_bin].append((current_position, item))
                current_position += item

        # Filter out bins with no items
        bins_with_items = [bin_idx for bin_idx, items_in_bin in enumerate(item_assignment) if items_in_bin]

        # Include a sample of unused bins (if any)
        unused_bins = [bin_idx for bin_idx, items_in_bin in enumerate(item_assignment) if not items_in_bin]
        if unused_bins:
            unused_bins_sample = unused_bins[:max_unused_bins]  # Sample a subset of unused bins
            bins_to_plot = bins_with_items + unused_bins_sample
        else:
            bins_to_plot = bins_with_items

        bins_to_plot = bins_to_plot[:n_show]

        # Adjust figure size based on the number of bins to plot
        bin_height = 0.5  # Height per bin in inches
        fig_height = max(3, len(bins_to_plot) * bin_height)  # Minimum height of 3 inches

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, fig_height))

        # Plot each bin and its items
        for plot_idx, bin_idx in enumerate(bins_to_plot):
            # Plot the bin as a horizontal bar
            ax.barh(plot_idx, capacity, height=0.6, color='lightgray', edgecolor='black', label='Bin' if plot_idx == 0 else None)

            # Plot the items packed into the bin (if any)
            for position, item in item_assignment[bin_idx]:
                ax.barh(plot_idx, item, left=position, height=0.6, color='skyblue', edgecolor='black')

        # Set axis labels and title
        ax.set_yticks(range(len(bins_to_plot)))
        ax.set_yticklabels([f'Bin {bin_idx + 1}' for bin_idx in bins_to_plot])
        ax.set_xlabel('Capacity')
        ax.set_title('1D Online Bin Packing Solution')

        # Add a legend
        ax.legend(['Bin', 'Item'], loc='upper right')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Show the plot
        plt.show()

    def get_valid_bin_indices(self, item: float, bins: np.ndarray) -> np.ndarray:
        """Returns indices of bins in which item can fit."""
        return np.nonzero((bins - item) >= 0)[0]

    def online_binpack(self,
                       items: tuple[float, ...], bins: np.ndarray, priority: callable
                       ) -> tuple[list[list[float, ...], ...], np.ndarray]:
        """Performs online binpacking of `items` into `bins`."""
        # Track which items are added to each bin.
        packing = [[] for _ in bins]
        # Add items to bins.
        for item in items:
            # Extract bins that have sufficient space to fit item.
            valid_bin_indices = self.get_valid_bin_indices(item, bins)
            # Score each bin based on heuristic.
            priorities = priority(item, bins[valid_bin_indices])
            # Add item to bin with highest priority.
            best_bin = valid_bin_indices[np.argmax(priorities)]
            bins[best_bin] -= item
            packing[best_bin].append(item)
        # Remove unused bins from packing.
        packing = [bin_items for bin_items in packing if bin_items]
        return packing, bins

    def evaluate(self, priority: callable) -> float:
        num_bins = []
        total_eval_time = 0  # 🌟 Thêm biến để tính tổng thời gian
        for name in self._datasets:
            instance = self._datasets[name]
            capacity = instance['capacity']
            items = instance['items']
            bins = np.array([capacity for _ in range(instance['num_items'])])

            start_time = time.time()  # 🌟 Ghi lại thời gian bắt đầu
            _, bins_packed = self.online_binpack(items, bins, priority)
            end_time = time.time()  # 🌟 Ghi lại thời gian kết thúc

            total_eval_time += end_time - start_time  # 🌟 Cộng dồn thời gian chạy của mỗi instance
            num_bins.append((bins_packed != capacity).sum())

        avg_num_bins = np.mean(num_bins)
        avg_eval_time = total_eval_time / self.n_instances  # 🌟 Tính thời gian trung bình
        # 🌟 Trả về một tuple hoặc mảng chứa cả hiệu suất và thời gian
        return np.array([-avg_num_bins, -avg_eval_time]), avg_eval_time


if __name__ == '__main__':
    def priority(item: float, valid_bins: np.ndarray) -> np.ndarray:
        """
        Priority function for the First-Fit Decreasing (FFD) heuristic.

        Args:
            item: The size of the item to be packed.
            valid_bins: A numpy array of remaining capacities in valid bins.

        Returns:
            A numpy array of priorities for the valid bins.
        """
        # Prioritize bins with the least remaining capacity (but still able to fit the item)
        priorities = -valid_bins  # Negative because we want to maximize the priority for the smallest remaining capacity
        return priorities


    obp = OBPEvaluation()
    ave_bins = obp.evaluate_program('_', priority)
    print(ave_bins)
