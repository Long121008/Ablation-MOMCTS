from analysis.IGD import plot_igd, build_reference_pf
from analysis.HV import calculate_hv_progression
from plot_pareto_front import compare_pareto_from_algorithms
from utils import read_json


def run_analysis(
    metric,
    problem,
    igd_ylim=None,         
    exclude_algorithms=None
):
   
    config = read_json("analysis/analysis_problem_test_size_100.json")
    algorithms = config[problem]

    if exclude_algorithms:
        algorithms = {k: v for k, v in algorithms.items() if k not in exclude_algorithms}

    if metric == "hv":
        calculate_hv_progression(
            algorithms,
            batch_size=10,
            visualize=True,
            max_samples=300,
            print_detail=True,
        )

    elif metric == "igd":
        all_paths = [p for paths in algorithms.values() for p in paths]
        ref_pf = build_reference_pf(all_paths)

        plot_igd(
            algorithms,
            ref_pf,
            max_eval=300,
            step=10,
            ylim=igd_ylim,   
            print_final=True
        )

    elif metric == "pareto":
        compare_pareto_from_algorithms(
            algorithms
        )

    else:
        raise ValueError(f"Unknown metric: {metric}")


if __name__ == "__main__":

    run_analysis(
        metric="igd",
        problem="bi_cvrp",
        igd_ylim=(0.0, 15.0),
        exclude_algorithms=[]  
    )