from __future__ import annotations

from smac.intensifier.successive_halving import SuccessiveHalving


def compute_hyperband_budgets(
        b_min: float,
        b_max: float,
        eta: int,
        n_stages: int | None = None,
        print_output: bool = True,
        is_prior_band: bool = False
    ) -> tuple[dict[int, list], dict[int, list], dict[int, list], int, float, float]:
    s_max = SuccessiveHalving._get_max_iterations(eta, b_max, b_min)

    n_configs_in_stage = {}
    budgets_in_stage = {}

    for i in range(s_max + 1):
        max_iter = s_max - i

        budgets_in_stage[i], n_configs_in_stage[i] = SuccessiveHalving._compute_configs_and_budgets_for_stages(
            eta, b_max, max_iter, s_max
        )

        for j in range(len(n_configs_in_stage[i])):
            if n_configs_in_stage[i][j] == 0:
                n_configs_in_stage[i][j] = 1

    total_real_budget = 0
    total_budget = 0

    total_trials = 0
    total_configs = 0

    # Real budgets using checkpointing
    real_budgets_in_stage = {}
    for i in range(len(budgets_in_stage)):
        # First budgets stays the same
        real_budgets_in_stage[i] = [budgets_in_stage[i][0]]

        for prev_budget, budget in zip(budgets_in_stage[i][:-1], budgets_in_stage[i][1:], strict=False):
            # We only train the difference
            real_budgets_in_stage[i].append(budget - prev_budget)

    # Now select how many stages we want
    if n_stages is None:
        n_stages = s_max + 1

    if is_prior_band:
        total_real_budget += b_max
        total_budget += b_max

        # We start by evaluationg the default configuration
        n_configs_in_stage[0] = [1] + n_configs_in_stage[0]
        budgets_in_stage[0] = [b_max] + budgets_in_stage[0]
        real_budgets_in_stage[0] = [b_max] + real_budgets_in_stage[0]

    for i in range(n_stages):
        configs_list = "[" + ", ".join([str(c).rjust(8) for c in n_configs_in_stage[i]]) + "]"
        budgets_list = "[" + ", ".join([str(round(b, 3)).rjust(8) for b in budgets_in_stage[i]]) + "]"
        real_budgets_list = "[" + ", ".join([str(round(b, 3)).rjust(8) for b in real_budgets_in_stage[i]]) + "]"

        stage_budget = sum([n * b for n, b in zip(n_configs_in_stage[i], budgets_in_stage[i], strict=False)])
        stage_real_budget = sum([n * b for n, b in zip(n_configs_in_stage[i], real_budgets_in_stage[i], strict=False)])

        total_trials += sum(n_configs_in_stage[i])

        if is_prior_band and i == 0:
            total_configs += n_configs_in_stage[i][1]
        else:
            total_configs += n_configs_in_stage[i][0]
        total_budget += stage_budget
        total_real_budget += stage_real_budget

        if print_output:
            print("-" * 80)
            print(f"Stage {i}: Budget {round(stage_budget, 3)}, Real Budget {round(stage_real_budget, 3)}")
            print(f"  #Configs:     {configs_list}")
            print(f"  Budgets:      {budgets_list}")
            print(f"  Real Budgets: {real_budgets_list}")
            print("-" * 80)

    if print_output:
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
        for n_config, real_budget in zip(n_configs, real_budgets_in_stage[stage], strict=False):
            for _ in range(n_config):
                real_budget_per_config[config_id] = real_budget
                config_id += 1

    return real_budget_per_config