import json
import numpy as np
import os
import re
import glob
from pathlib import Path
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def read_json(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def read_score_from_path(json_data: str) -> list[list[float, float]]:
    skip_item_num = 0
    with open(json_data, "r") as f:
        data = json.load(f)
    scores = []
    for item in data:
        if 'score' in item and isinstance(item['score'], list) and len(item['score']) == 2:
            scores.append(item['score'])
        else:
            skip_item_num += 1
            print(f"Warning: Skipping item due to invalid 'score'.")
    print(f"Skip item numbers: {skip_item_num}")
    return scores

def find_pareto_front_from_scores(scores: list[list[float, float]]):
    F_hist_np = np.array(scores)
    nd_indices = NonDominatedSorting().do(F_hist_np, only_non_dominated_front=True)
    true_pf_approx = F_hist_np[nd_indices]

    return true_pf_approx


def read_population_scores_from_folder(folder_path: str) -> list[list[float, float]]:
    '''
    Args:
        mark = 0: the score is negative
        mark = 1: objective is positive
    '''
    mark = 0
    files = glob.glob(os.path.join(folder_path, "pop_*.json"))
    if len(files) == 0:
        mark = 1
        files = glob.glob(os.path.join(folder_path, "population_generation_*.json"))
        
    files.sort(key=lambda x: int(re.search(r"(\d+)", os.path.basename(x)).group()))
    data_list = []

    for file_path in files:
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = None  # or {}
            data_list.append({
                "filename": os.path.basename(file_path),
                "content": data
            })

    F_list = []
    for data in data_list:
        F = []
        for x in data["content"]:
            if mark == 0:
                obj, runtime = x["score"]
            else:
                obj, runtime = x["score"]
            F.append([obj, runtime])
        F_list.append(F)

    return F_list

def calculate_true_pareto_front(folder_list: list[str]) -> np.ndarray:
    full_scores = []
    file_path_name = ["samples_1~200.json", "samples_0~200.json", "samples_1~300.json", "samples_0~300.json"]
    for folder in folder_list:
        folder_path = Path(folder)
        for name in file_path_name:
            for file_path in folder_path.rglob(name):  # recursive search
                try:
                    print(f"Get from file path: {file_path}")
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    scores = [item.get("score") for item in data if item.get("score") is not None]
                    scores = [[x for x in pair] for pair in scores]
                    full_scores.extend(scores)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    F_hist_np = np.array(full_scores)
    true_nd_indices = NonDominatedSorting().do(F_hist_np, only_non_dominated_front=True)
    true_pf_approx = F_hist_np[true_nd_indices]  # get true Pareto front
    
    return true_pf_approx