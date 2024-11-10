from __future__ import annotations
import math
from collections import defaultdict
from smac.intensifier.successive_halving import SuccessiveHalving
import numpy as np

def compute_successive_halving(
        min_budget: float,
        max_budget: float,
        eta: float,
        early_stopping_rate: int = 0,
    ):
    stopping_rate_limit = np.floor(
            np.log(max_budget / min_budget) / np.log(eta)
        ).astype(int)
    
    new_min_budget = min_budget * (eta**early_stopping_rate)
    nrungs = (
        np.floor(np.log(max_budget / new_min_budget) / np.log(eta)).astype(
            int
        )
        + 1
    )
    
    # Rung Map
    _max_budget = max_budget
    rung_map = dict()
    for i in reversed(range(nrungs)):
        rung_map[i + early_stopping_rate] = _max_budget
        _max_budget /= eta

    # Config Map
    s_max = stopping_rate_limit + 1
    _s = stopping_rate_limit - early_stopping_rate
    _n_config = np.floor(s_max / (_s + 1)) * eta**_s
    config_map = dict()
    for i in range(nrungs):
        config_map[i + early_stopping_rate] = int(_n_config)
        _n_config //= eta

    n_configs = []
    budgets = []

    for i in sorted(rung_map.keys()):
        n_configs.append(config_map[i])
        budgets.append(rung_map[i])

    return n_configs, budgets

def compute_hyperband_brackets(
        b_min: float,
        b_max: float,
        eta: int
    ) -> tuple[int, dict[int, list], dict[int, list]]:
    s_max = math.floor(math.log(b_max / b_min) / math.log(eta))
    n_configs_in_stage = defaultdict(list)
    budgets_in_stage = defaultdict(list)

    for s in range(s_max + 1):
        n_configs_in_stage[s], budgets_in_stage[s] = compute_successive_halving(b_min, b_max, eta, early_stopping_rate=s)

    return s_max, n_configs_in_stage, budgets_in_stage

def compute_hyperband_budgets(
        b_min: float,
        b_max: float,
        eta: int,
        n_stages: int | None = None,
        print_output: bool = True,
        is_prior_band: bool = False
    ) -> tuple[dict[int, list], dict[int, list], dict[int, list], int, float, float]:
    s_max, n_configs_in_stage, budgets_in_stage = compute_hyperband_brackets(b_min, b_max, eta)



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

        # In hyperband, only the first budget in each stages contains new configurations.
        # In PriorBand, we have to check whether this is the default configuration (i=0)
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


def get_budget_per_config(
        n_configs_in_stage: dict[int, list],
        budgets_in_stage: dict[int, list],
    ) -> dict[int, float]:
    budget_per_config = {}

    config_id = 0
    for stage, n_configs in n_configs_in_stage.items():
        for n_config, real_budget in zip(n_configs, budgets_in_stage[stage], strict=False):
            for _ in range(n_config):
                budget_per_config[config_id] = real_budget
                config_id += 1

    return budget_per_config