import numpy as np
import matplotlib.pyplot as plt
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from utils import read_score_from_path
from analysis.plot_style import PAPER_STYLES, STD_ALPHA


def normalize(F, ideal, nadir):
    return (F - ideal) / (nadir - ideal + 1e-12)

def extract_final_igd(curves, max_eval):
    
    finals = []
    for c in curves:
        # take the largest eval <= max_eval
        valid_steps = [e for e in c.keys() if e <= max_eval]
        if not valid_steps:
            continue
        last_step = max(valid_steps)
        val = c[last_step]
        if not np.isnan(val):
            finals.append(val)

    if len(finals) == 0:
        return np.nan, np.nan

    return np.mean(finals), np.std(finals)

def build_reference_pf(all_paths):
    all_F = []
    for p in all_paths:
        all_F.extend(read_score_from_path(p))
    all_F = np.asarray(all_F)

    nd_idx = NonDominatedSorting().do(all_F, only_non_dominated_front=True)
    return all_F[nd_idx]


def igd_curve(path, ref_pf, ideal, nadir, max_eval=300, step=10):
    F = np.asarray(read_score_from_path(path))
    if len(F) == 0:
        return {}

    ref_pf = normalize(ref_pf, ideal, nadir)
    metric = IGD(ref_pf)

    curve = {}
    for t in range(0, max_eval + 1, step):
        A = F[:t + 1]
        if len(A) == 0:
            curve[t] = np.nan
            continue

        nd = NonDominatedSorting().do(A, only_non_dominated_front=True)
        P = A[nd]
        curve[t] = metric.do(normalize(P, ideal, nadir)) if len(P) > 0 else np.nan

    return curve


def aggregate_igd(paths, ref_pf, max_eval=300, step=10, return_curves=False):
    ideal = ref_pf.min(axis=0)
    nadir = ref_pf.max(axis=0)

    curves = [igd_curve(p, ref_pf, ideal, nadir, max_eval, step) for p in paths]
    curves = [c for c in curves if c]

    evals = sorted(set().union(*[c.keys() for c in curves]))
    vals = np.array([[c.get(e, np.nan) for e in evals] for c in curves])

    mean = np.nanmean(vals, axis=0)
    std = np.nanstd(vals, axis=0)

    if return_curves:
        return evals, mean, std, curves
    else:
        return evals, mean, std



def plot_igd(
    algorithms,
    ref_pf,
    max_eval=300,
    step=10,
    ylim=None,
    print_final=True,
    precision=4
):
    plt.figure(figsize=(7, 5))

    final_results = {}

    for algo, paths in algorithms.items():
        x, mean, std, curves = aggregate_igd(
            paths, ref_pf, max_eval, step, return_curves=True
        )
        if len(x) == 0:
            continue

        style = PAPER_STYLES.get(algo, {})
        plt.plot(x, mean, label=algo, **style)
        plt.fill_between(
            x,
            mean - std,
            mean + std,
            color=style.get("color", None),
            alpha=STD_ALPHA
        )

        # ---- final IGD ----
        if print_final:
            m, s = extract_final_igd(curves, max_eval)
            final_results[algo] = (m, s)

    plt.xlabel("Function evaluations", fontsize=16)
    plt.title(r"IGD $\downarrow$", fontsize=18)

    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=14)

    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)

    leg = plt.legend(
        loc="upper right",
        fontsize=14,
        frameon=True,
        framealpha=0.95,
        handlelength=3.0,
        handletextpad=0.8,
        borderpad=0.8,
    )

    for line in leg.get_lines():
        line.set_linewidth(2.5)

    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    plt.show()

    if print_final and final_results:
        print("\nFinal IGD (mean ± std):")
        for algo, (m, s) in final_results.items():
            if np.isnan(m):
                print(f"{algo:20s}: N/A")
            else:
                print(
                    f"{algo:20s}: "
                    f"{m:.{precision}f} ± {s:.{precision}f}"
                )
