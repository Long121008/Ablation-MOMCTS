from HV_func_eval import calculate_hv_progression
from IGD_func_eval import compare_igd_curves_multi
from utils import calculate_true_pareto_front, read_json
from plot_pareto_front import compare_pareto_from_algorithms


def run_analysis(metric="pareto", problem="tsp_semo", working_dir = None):

    problem_dict = read_json("analysis/analysis_problem_test_size_100.json")
    algorithms = problem_dict[problem]

    if metric == "hv":
        calculate_hv_progression(algorithms, batch_size=10)

    elif metric == "igd":
        true_pf_approx = calculate_true_pareto_front([
            f"logs/momcts/{problem}/{working_dir}",
            f"logs/meoh/{problem}/{working_dir}",
            f"logs/nsga2/{problem}/{working_dir}",
            f"logs/moead/{problem}/{working_dir}"
        ])
        compare_igd_curves_multi(algorithms, true_pf_approx, max_eval=300)

    elif metric == "pareto":
        compare_pareto_from_algorithms(algorithms, show_global=True)

    else:
        raise ValueError(f"Unknown metric: {metric}")


if __name__ == "__main__":
    run_analysis(metric="hv", problem="bi_kp", working_dir="test_100")
