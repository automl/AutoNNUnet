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


def non_dominated_sorting(configs: dict[str, tuple[float]]) -> list[list[str]]:
    """Get the non-dominated configurations."""
    population = configs.copy()
    fronts = []  

    while population:
        current_front = [] 

        for config_id, config_result in population.items():
            dominated = False
            for other_id, other_config_result in population.items():
                if config_id == other_id:
                    continue

                performance = list(config_result)
                other_performance = list(other_config_result)

                if (other_performance[0] <= performance[0] and other_performance[1] <= performance[1]) and (
                        other_performance[0] < performance[0] or other_performance[1] < performance[1]):
                    dominated = True
                    break

            if not dominated:
                current_front.append(config_id)

        fronts.append(current_front)

        for config_id in current_front:
            population.pop(config_id)

    return fronts

def compute_distance(result1: tuple[float], result2: tuple[float]) -> float:
    """Compute the distance between two results."""
    r1 = np.array(result1)
    r2 = np.array(result2)

    return float(np.linalg.norm(r1 - r2))


def crowding_distance_assignment(configs: dict[str, tuple[float]]) -> dict[str, float]:
    distances = {c: 0. for c in configs}
    objective_bounds = {
        0: [0, 1],
        1: [0, 1]
    }

    for o in objective_bounds.keys(): 
        o_min, o_max = objective_bounds[o]
        sorted_configs = sorted(configs.keys(), key=lambda c: configs[c][o])
        distances[sorted_configs[0]] = np.inf
        distances[sorted_configs[-1]] = np.inf

        for i in range(2, len(sorted_configs) - 1):
            config_key = sorted_configs[i]
            distances[config_key] += (configs[sorted_configs[i + 1]][o] - configs[sorted_configs[i + 1]][o]) / (o_max - o_min)

    return distances    


def eps_net(performances_in_stage: dict[str, tuple[float]]):
    fronts = non_dominated_sorting(performances_in_stage)

    sorted_configs = [] 

    # We start by selecting the first element of the first front
    sorted_configs.append(fronts[0].pop(0))  

    # Iterate over each front
    for front in fronts:
        while front:
            # Find the configuration in the current that maximizes the minimum distance to the current set C
            max_min_distance = -np.inf
            best_config_id = None

            for config_id in front:
                # Compute the minimum distance between the current config and each config in C
                distances = [compute_distance(performances_in_stage[config_id], performances_in_stage[c]) for c in sorted_configs]

                min_distance = min(distances)
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_config_id = config_id

            # Add the selected configuration to C and remove it from F
            if best_config_id is not None:
                sorted_configs.append(best_config_id)
                front.remove(best_config_id)

    return sorted_configs


def get_mockup_moo_points(b_min: float = 10, b_max: float = 1000, eta: int = 3) -> tuple[pd.DataFrame, dict]:
    """Generate mockup data for a multi-objective hyperband run."""
    n_configs_in_stage, budgets_in_stage, _, _, _, _ = compute_hyperband_budgets(
        b_min=b_min,
        b_max=b_max,
        eta=eta,
        print_output=False
    )

    def get_performance_for_budget(budget: float) -> float:
        return np.random.uniform(budget / b_max, 1)

    performances_in_stage = defaultdict(lambda: defaultdict(dict))

    for stage in n_configs_in_stage:
        for phase, (n_configs, budget) in enumerate(zip(n_configs_in_stage[stage], budgets_in_stage[stage])):
            if phase == 0:
                # Here we have to "sample" configurations
                for config_id in range(n_configs):
                    performances_in_stage[stage][phase][config_id] = (
                        get_performance_for_budget(budget),
                        get_performance_for_budget(budget)
                    )
            else:
                # Now we select the best configurations from the previous stage
                # and add some noise to them
                previous_performances = performances_in_stage[stage][phase - 1]
                
                fronts = non_dominated_sorting(previous_performances)
                selected_configs = []

                # First we start by adding full pareto fronts
                while True:
                    if len(selected_configs) + len(fronts[0]) <= n_configs:
                        # We can add the full front
                        selected_configs.extend(fronts.pop(0))
                    else:
                        break
                
                # Now, we need to selct the remaining configs from the next front
                remaining_front = {c: previous_performances[c] for c in fronts[0]}
                crowding_distance = crowding_distance_assignment(remaining_front)
                sorted_configs = sorted(crowding_distance.keys(), key=lambda c: (crowding_distance[c], compute_hypervolume(previous_performances[c])))

                selected_configs.extend(sorted_configs[:n_configs - len(selected_configs)])

                for config_id in selected_configs:
                    performances_in_stage[stage][phase][config_id] = (
                        previous_performances[config_id][0] - np.random.uniform(budget / b_max, previous_performances[config_id][0]),
                        previous_performances[config_id][1] - np.random.uniform(budget / b_max, previous_performances[config_id][1]),
                    )

    # Transform into a dataframe with columns [config_id, performance_1, performance_2, stage, phase]
    data = []
    for stage, performances in performances_in_stage.items():
        for phase, configs in performances.items():
            for config_id, performance in configs.items():
                data.append([config_id, performance[0], performance[1], stage, phase])

    performance_data = pd.DataFrame(
        data=data,
        columns=["Configuration ID", "Objective 1", "Objective 2", "Stage", "Phase"]
    )

    return performance_data, performances_in_stage


def compute_hypervolume(performance: tuple[float]) -> float:
    r = np.array((-0.1, -0.1))
    p = np.array(performance)

    return float(np.prod(p - r))


def plot_mockup_moo_points(
        performance_data: pd.DataFrame,
        performances_in_stage: dict,
        stage: int = 0,
        step: int = 0
    ) -> None:
    """Plot the mockup data for a multi-objective hyperband run."""
    performance_stage = performance_data[performance_data["Stage"] == stage]
    n_phases = len(performance_stage["Phase"].unique())

    fig, axes = plt.subplots(
        1,
        n_phases - 1,
        figsize=(12, 6),
        sharex=True,
        sharey=True
    )

    axes = axes.flatten()

    for ax, phase in zip(axes, range(1, n_phases)):
        # ---------------------------------------------------------------------
        # Plot the data points
        # ---------------------------------------------------------------------
        for cur_phase in range(0, phase + 1):
            if step <= 1 and cur_phase == phase:
                continue

            sns.scatterplot(
                data=performance_stage[performance_stage["Phase"] == cur_phase],
                x="Objective 1",
                y="Objective 2",
                color=sns.color_palette()[cur_phase],
                ax=ax,
                alpha=0.5 if cur_phase < phase - 1 else 1
            )

        ax.set_title(f"Phase {phase}")

        # ---------------------------------------------------------------------
        # Plot the selection strategy
        # ---------------------------------------------------------------------
        if step >= 1:
            fronts = non_dominated_sorting(performances_in_stage[stage][phase - 1])
            sorted_configs = eps_net(performances_in_stage[stage][phase - 1])

            prev_phase_data = performance_stage[performance_stage["Phase"] == phase - 1]

            for i, front in enumerate(fronts):
                front_configs = [config for config in front if config in sorted_configs]
                front_data = prev_phase_data[prev_phase_data["Configuration ID"].isin(front_configs)]

                front_data = front_data.sort_values(by=["Objective 1", "Objective 2"])

                # dotted line
                ax.plot(
                    front_data["Objective 1"],
                    front_data["Objective 2"],
                    color="black",
                    lw=2,
                    alpha=0.5,
                    ls="--"
                )

        # ---------------------------------------------------------------------
        # Plot the arrows
        # ---------------------------------------------------------------------
        if step >= 2:
            this_phase = performance_stage[performance_stage["Phase"] == phase]
            for _, row in this_phase.iterrows():
                config_id = row["Configuration ID"]
                performance = (row["Objective 1"], row["Objective 2"])

                # Now we find the previous performance
                prev_phase = performance_stage[performance_stage["Phase"] == phase - 1]
                prev_row = prev_phase[prev_phase["Configuration ID"] == config_id]
                prev_performance = (prev_row["Objective 1"].values[0], prev_row["Objective 2"].values[0])

                ax.annotate(
                    "",
                    xy=performance,
                    xytext=prev_performance,
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=2,
                        color="black"
                    )
                )

    # Add a legend for the phases
    handles, labels = [], []
    for phase in range(n_phases):
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=sns.color_palette()[phase], markersize=10))
        labels.append(phase)

    axes[-2].legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=len(performance_stage["Phase"].unique()),
        fancybox=False,
        shadow=False,
        frameon=False,
        title="Phase"
    )

    axes[-2].set_ylim(-0.6, 1.05)
    axes[-2].set_yticks([-0.5, 0, 0.5, 1])
    axes[-2].set_xlim(-0.75, 1.05)
    axes[-2].set_xticks([-0.5, 0, 0.5, 1])
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"mo_pb_selection_strategy_{step}.png", dpi=400)


if __name__ == "__main__":
    performance_data, performances_in_stage = get_mockup_moo_points()
    for step in range(3):
        plot_mockup_moo_points(performance_data, performances_in_stage, stage=1, step=step)