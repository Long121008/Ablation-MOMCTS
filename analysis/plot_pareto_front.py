import numpy as np
import matplotlib.pyplot as plt
from utils import read_score_from_path, find_pareto_front_from_scores


def compare_pareto_from_algorithms(
    file_dict: dict[str, list[str]],
    highlight_algo: str = "MOMCTS",
):
    """
    Paper-quality Pareto front comparison.
    MOMCTS is visually highlighted; baselines are de-emphasized.
    """

    # =========================
    # Figure setup
    # =========================
    plt.figure(figsize=(7.5, 5.5))

    COLORS = {
        "MOMCTS": "#1f77b4",   # blue (highlight)
        "NSGA2":  "#2ca02c",   # green
        "MOEAD":  "#9467bd",   # purple
        "MPAGE":  "#ff7f0e",   # orange
        "MEOH":   "#d62728",   # red
    }

    # =========================
    # Local Pareto fronts
    # =========================
    for algo, file_list in file_dict.items():
        scores_all = []

        for path in file_list:
            scores = read_score_from_path(path)
            if len(scores) > 0:
                scores_all.extend(scores)

        if not scores_all:
            print(f"⚠️ No valid data for {algo}")
            continue

        scores_all = np.asarray(scores_all, dtype=float)
        pareto = find_pareto_front_from_scores(scores_all)

        # Sort for clean Pareto curve
        pareto = pareto[np.argsort(pareto[:, 0])]

        is_highlight = highlight_algo.lower() in algo.lower()
        color = COLORS.get(algo, "#999999")

        # =========================
        # Plotting
        # =========================
        if is_highlight:
            # --- MOMCTS (dominant) ---
            plt.plot(
                pareto[:, 0],
                pareto[:, 1],
                color=color,
                linewidth=2.8,
                label=f"{algo} (Pareto Front)",
                zorder=5,
            )
            plt.scatter(
                pareto[:, 0],
                pareto[:, 1],
                s=110,
                color=color,
                edgecolor="black",
                linewidth=1.2,
                zorder=6,
            )
        else:
            # --- Baselines (context only) ---
            plt.scatter(
                pareto[:, 0],
                pareto[:, 1],
                s=55,
                color=color,
                alpha=0.45,
                label=algo,
                zorder=2,
            )

    # =========================
    # Axis & styling
    # =========================
    plt.xlabel("− Hypervolume (↑ better)", fontsize=12)
    plt.ylabel("Runtime (↓ better)", fontsize=12)

    plt.xlim(-1.0, -0.5)
    plt.ylim(0.0, 5)

    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

    plt.legend(
        fontsize=10,
        frameon=True,
        loc="upper right",
    )

    plt.title("Pareto Front Comparison Across Algorithms", fontsize=13)

    plt.tight_layout()
    plt.show()
