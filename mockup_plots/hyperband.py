import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any
from autonnunet.utils.hyperband import compute_hyperband_budgets
from collections import defaultdict
from pathlib import Path


np.random.seed(42)

sns.set_style(style="darkgrid")
sns.set_palette(palette="colorblind")


PLOTS_DIR = Path("./mockup_plots").resolve()


def get_mockup_points(b_min: float = 10, b_max: float = 1000, eta: int = 3) -> tuple[pd.DataFrame, dict]:
    """Generate mockup data for a multi-objective hyperband run."""
    n_configs_in_stage, budgets_in_stage, _, _, _, _ = compute_hyperband_budgets(
        b_min=b_min,
        b_max=b_max,
        eta=eta,
        print_output=False
    )

    def get_performance_for_budget(budget: float) -> float:
        return np.random.uniform(1 - (budget / b_max), 1)

    performances_in_stage = defaultdict(lambda: defaultdict(dict))

    for stage in n_configs_in_stage:
        for phase, (n_configs, budget) in enumerate(zip(n_configs_in_stage[stage], budgets_in_stage[stage])):
            if phase == 0:
                # Here we have to "sample" configurations
                for config_id in range(n_configs):
                    performances_in_stage[stage][phase][config_id] = get_performance_for_budget(budget)
            else:
                # Now we select the best configurations from the previous stage
                # and add some noise to them
                previous_performances = performances_in_stage[stage][phase - 1]
                
                # We select the top performing configs
                selected_configs = sorted(previous_performances, key=lambda x: previous_performances[x])[:n_configs]

                for config_id in selected_configs:
                    performances_in_stage[stage][phase][config_id] = np.random.uniform(1 - (budget / b_max), previous_performances[config_id])

    # Transform into a dataframe with columns [config_id, performance_1, performance_2, stage, phase]
    data = []
    for stage, performances in performances_in_stage.items():
        for phase, configs in performances.items():
            for config_id, performance in configs.items():
                budget = budgets_in_stage[stage][phase]
                data.append([config_id, budget, performance, stage, phase])

    performance_data = pd.DataFrame(
        data=data,
        columns=["Configuration ID", "Budget", "Loss", "Stage", "Phase"]
    )

    return performance_data, performances_in_stage


def plot_mockup_points(
        performance_data: pd.DataFrame,
        performances_in_stage: dict,
    ) -> None:
    """Plot the mockup data for a multi-objective hyperband run."""
    n_stages = len(performance_data["Stage"].unique())
    budgets = np.unique(performance_data["Budget"])

    fig, axes = plt.subplots(
        1,
        n_stages - 2,
        figsize=(8, 2),
    )

    axes = axes.flatten()

    for ax, stage in zip(axes, range(0, n_stages - 2)):
        performance_stage = performance_data[performance_data["Stage"] == stage]

        # ---------------------------------------------------------------------
        # Plot the lines showing successive halving
        # ---------------------------------------------------------------------
        n_phases = len(performance_stage["Phase"].unique())

        for cur_phase in range(0, n_phases):
            performance_phase = performance_stage[performance_stage["Phase"] == cur_phase]
            performance_prev_phase = performance_stage[performance_stage["Phase"] == cur_phase - 1]

            if cur_phase == 0:
                prev_budget = {config_id: 0.01 for config_id in performance_phase["Configuration ID"].values}
                prev_loss = {config_id: 1 for config_id in performance_phase["Configuration ID"].values}

            else:
                prev_budget = {config_id: performance_prev_phase[performance_prev_phase["Configuration ID"] == config_id]["Budget"].values[0] for config_id in performance_phase["Configuration ID"].values}
                prev_loss = {config_id: performance_prev_phase[performance_prev_phase["Configuration ID"] == config_id]["Loss"].values[0] for config_id in performance_phase["Configuration ID"].values}

            # Plot lineplot for every configuration from prev phase to current phase
            for i, (_, row) in enumerate(performance_phase.iterrows()):
                
                # We want to highlight configurations that are selected for the next phase
                alpha = 1 / (len(performances_in_stage[stage][cur_phase]))
                if cur_phase == n_phases - 1 or row["Configuration ID"] in performances_in_stage[stage][cur_phase + 1]:
                    alpha = 1

                budget_idx = np.where(budgets == row["Budget"])[0][0]

                ax.plot(
                    [prev_budget[row["Configuration ID"]], row["Budget"]],
                    [prev_loss[row["Configuration ID"]], row["Loss"]],
                    color=sns.color_palette("colorblind")[budget_idx],
                    alpha=alpha,
                )

        ax.set_xscale("log")
        ax.set_xticks(np.unique(performance_stage["Budget"]))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("Budget")

    # We use the budgets and colors from palette as handles and labels
    handles = [plt.Line2D([0], [0], color=sns.color_palette("colorblind")[i], linewidth=3) for i in range(len(budgets))]
    labels = [str(round(v)) for v in budgets]

    # adjust spacing between subplots
    fig.subplots_adjust(
        top=0.96,   
        bottom=0.21, 
        left=0.04,  
        right=0.99,  
        wspace=0.1  
    )

    axes[-2].legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=5,
        fancybox=False,
        shadow=False,
        frameon=False,
    )

    axes[0].set_ylabel("Loss")
    
    plt.savefig(PLOTS_DIR / f"hyperband.png", dpi=400)


if __name__ == "__main__":
    performance_data, performances_in_stage = get_mockup_points()
    plot_mockup_points(performance_data, performances_in_stage)