import numpy as np
from smac.intensifier.successive_halving import SuccessiveHalving


def compute_hyperband_budgets(b_min, b_max, eta) -> tuple[dict[int, list], dict[int, list], int, int]:
    s_max = SuccessiveHalving._get_max_iterations(eta, b_max, b_min)

    n_configs_in_stage = {}
    budgets_in_stage = {}

    for i in range(s_max + 1):
        max_iter = s_max - i

        budgets_in_stage[i], n_configs_in_stage[i] = SuccessiveHalving._compute_configs_and_budgets_for_stages(
            eta, b_max, max_iter, s_max
        )
        budgets_in_stage[i] = [round(b) for b in budgets_in_stage[i]]

    total_budget = 0
    for i in range(s_max + 1):
        for n_configs, budget in zip(n_configs_in_stage[i], budgets_in_stage[i], strict=False):
            total_budget += n_configs * budget

    total_trials = np.sum([np.sum(v) for v in n_configs_in_stage.values()])

    print("-" * 80)
    for i in range(s_max + 1):
        configs_list = "[" + ", ".join([str(c).rjust(4) for c in n_configs_in_stage[i]]) + "]"
        budgets_list = "[" + ", ".join([str(b).rjust(4) for b in budgets_in_stage[i]]) + "]"

        print(f"Stage {i}")
        print(f"  #Configs: {configs_list}")
        print(f"  Budgets:  {budgets_list}")
        print("-" * 80)

    print("Number of HB brackets: ", s_max + 1)
    print("Total number of trials: ", total_trials)
    print("Total budget: ", total_budget)

    return n_configs_in_stage, budgets_in_stage, total_trials, total_budget