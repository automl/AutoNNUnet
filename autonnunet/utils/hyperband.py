from __future__ import annotations

import numpy as np
from smac.intensifier.successive_halving import SuccessiveHalving


def compute_hyperband_budgets(
        b_min: float, 
        b_max: float,
        eta: int,
        print_output: bool = True
    ) -> tuple[dict[int, list], dict[int, list], dict[int, list], int, float, float]:
    s_max = SuccessiveHalving._get_max_iterations(eta, b_max, b_min)

    n_configs_in_stage = {}
    budgets_in_stage = {}

    for i in range(s_max + 1):
        max_iter = s_max - i

        budgets_in_stage[i], n_configs_in_stage[i] = SuccessiveHalving._compute_configs_and_budgets_for_stages(
            eta, b_max, max_iter, s_max
        )

    total_real_budget = 0
    total_budget = 0
    real_budgets_in_stage = {}

    # Real budgets using checkpointing
    for i in range(s_max + 1):
        # First budgets stays the same
        real_budgets_in_stage[i] = [budgets_in_stage[i][0]]

        for prev_budget, budget in zip(budgets_in_stage[i][:-1], budgets_in_stage[i][1:], strict=False):
            # We only train the difference
            real_budgets_in_stage[i].append(budget - prev_budget)

        # Now we can use the real budgets to compute the total budget
        for n_configs, budget, real_budget in zip(
            n_configs_in_stage[i],
            budgets_in_stage[i],
            real_budgets_in_stage[i],
            strict=False
        ):
            total_budget += n_configs * budget
            total_real_budget += n_configs * real_budget

    total_trials = np.sum([np.sum(v) for v in n_configs_in_stage.values()])
    total_configs = np.sum([v[0] for v in n_configs_in_stage.values()])

    if print_output:
        print("-" * 80)
        for i in range(s_max + 1):
            configs_list = "[" + ", ".join([str(c).rjust(8) for c in n_configs_in_stage[i]]) + "]"
            budgets_list = "[" + ", ".join([str(b).rjust(8) for b in budgets_in_stage[i]]) + "]"
            real_budgets_list = "[" + ", ".join([str(b).rjust(8) for b in real_budgets_in_stage[i]]) + "]"

            print(f"Stage {i}")
            print(f"  #Configs:     {configs_list}")
            print(f"  Budgets:      {budgets_list}")
            print(f"  Real Budgets: {real_budgets_list}")
            print("-" * 80)

        print("Number of HB brackets: ", s_max + 1)
        print("Total number of trials: ", total_trials)
        print("Total number of configs: ", total_configs)
        print("Total budget: ", total_budget)
        print("Total real budget: ", total_real_budget)

    return n_configs_in_stage, budgets_in_stage, real_budgets_in_stage, total_trials, total_budget, total_real_budget


def get_real_budget_per_config(
        n_configs_in_stage: dict[int, list],
        real_budgets_in_stage: dict[int, list],
    ) -> dict[int, float]:
    real_budget_per_config = {}

    config_id = 0
    for stage, n_configs in n_configs_in_stage.items():
        for n_config, real_budget in zip(n_configs, real_budgets_in_stage[stage]):
            for _ in range(n_config):
                real_budget_per_config[config_id] = real_budget
                config_id += 1

    return real_budget_per_config