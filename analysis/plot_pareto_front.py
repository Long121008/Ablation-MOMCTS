import json
import numpy as np
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from utils import read_score_from_path, find_pareto_front_from_scores
from pathlib import Path


def compare_pareto_from_algorithms(file_dict: dict[str, list[str]]):
 
    plt.figure(figsize=(12, 10)) 

    for i, (algo, file_list) in enumerate(file_dict.items()):
        all_scores = []

        # collect all scores from multiple runs
        for file_path in file_list:
            scores = read_score_from_path(file_path)
            all_scores.extend(scores)

        all_scores = np.array(all_scores) #[[6.]]
        
     
        pareto_front = find_pareto_front_from_scores(all_scores)
     
        plt.scatter(pareto_front[:, 0], pareto_front[:, 1],
                    label=f"{algo}",
                    s=100)


    plt.xlim(-1.1, -0.80)   # zoom in closer around Pareto solutions
    plt.ylim(-0.1, 4)      # focus on 0–3 runtime (most interesting region)

    plt.xlabel("Hypervolume (negative, minimize)")
    plt.ylabel("Runtime (positive, minimize)")
    plt.title("Pareto Front Comparison Across Algorithms")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
 

    algo_files = {
      "MoMCTS_Mixtral": [
        "logs/momcts/tsp_semo/codestral_v1/samples/samples_1~200.json",
        "logs/momcts/tsp_semo/codestral_v2/samples/samples_1~200.json"    ],
    
    "MEOH_Mixtral": [
       "logs/meoh/tsp_semo/codestral_v1_latest_sort/samples/samples_0~300.json",
       "logs/meoh/tsp_semo/20250929_134828/samples/samples_0~300.json"
    ],
     "NSGA2_Mixtral": [
         "logs/nsga2/tsp_semo/codestral_300_v1/samples/samples_0~300.json",
        "logs/nsga2/tsp_semo/codestral_300_v2/samples/samples_0~300.json",
        "logs/nsga2/tsp_semo/codestral_300_v3/samples/samples_0~300.json"
    ],
    "MPaGe": [
        "logs/mpage/tsp_semo/20250929_140752/samples/samples_0~300.json",
        "logs/mpage/tsp_semo/codestral_v1_300/samples/samples_0~300.json"
    ],
    "MoMCTS_Tune": [
        "logs/momcts/tsp_semo/codestral_v3_tune/samples/samples_1~300.json",
        "logs/momcts/tsp_semo/codestral_v1_tune/samples/samples_1~300.json",
        "logs/momcts/tsp_semo/codestral_v2_tune/samples/samples_1~300.json"
    ]
    }

    compare_pareto_from_algorithms(algo_files)


'''
TSP_SEMO:
"MoMCTS_Mixtral": [
        "logs/momcts/tsp_semo/codestral_v1/samples/samples_1~200.json",
        "logs/momcts/tsp_semo/codestral_v2/samples/samples_1~200.json"    ],
    
    "MEOH_Mixtral": [
       "logs/meoh/tsp_semo/codestral_v1_latest_sort/samples/samples_0~300.json",
       "logs/meoh/tsp_semo/20250929_134828/samples/samples_0~300.json"
    ],
     "NSGA2_Mixtral": [
         "logs/nsga2/tsp_semo/codestral_300_v1/samples/samples_0~300.json",
        "logs/nsga2/tsp_semo/codestral_300_v2/samples/samples_0~300.json",
        "logs/nsga2/tsp_semo/codestral_300_v3/samples/samples_0~300.json"
    ],
    "MPaGe": [
        "logs/mpage/tsp_semo/20250929_140752/samples/samples_0~300.json",
        "logs/mpage/tsp_semo/codestral_v1_300/samples/samples_0~300.json"
    ],
    "MoMCTS_Tune": [
        "logs/momcts/tsp_semo/codestral_v3_tune/samples/samples_1~300.json",
        "logs/momcts/tsp_semo/codestral_v1_tune/samples/samples_1~300.json",
        "logs/momcts/tsp_semo/codestral_v2_tune/samples/samples_1~300.json"
    ]

'''

'''
Online KP:


'''