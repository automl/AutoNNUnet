"""Hyperband utilities."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def compute_successive_halving(
        min_budget: float,
        max_budget: float,
        eta: float,
        early_stopping_rate: int = 0,
    ) -> tuple[list[int], list[float]]:
    """Compute the number of configurations and budgets for each stage in
    Successive Halving.

    Parameters
    ----------
    min_budget : float
        Minimum budget.

    max_budget : float
        Maximum budget.

    eta : float
        Reduction factor.

    early_stopping_rate : int, optional
        Early stopping rate. Defaults to 0.

    Returns:
    -------
    tuple[list[int], list[float]]
        Number of configurations and budgets for each stage.
    """
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
    rung_map = {}
    for i in reversed(range(nrungs)):
        rung_map[i + early_stopping_rate] = _max_budget
        _max_budget /= eta

    # Config Map
    s_max = stopping_rate_limit + 1
    _s = stopping_rate_limit - early_stopping_rate
    _n_config = np.floor(s_max / (_s + 1)) * eta**_s
    config_map = {}
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
    """Compute the number of configurations and budgets for each stage in HyperBand.

    Parameters
    ----------
    b_min : float
        Minimum budget.

    b_max : float
        Maximum budget.

    eta : int
        Reduction factor.

    Returns:
    -------
    tuple[int, dict[int, list], dict[int, list]]
        Number of brackets, number of configurations in each stage, and
        budgets in each stage.
    """
    s_max = math.floor(math.log(b_max / b_min) / math.log(eta))
    n_configs_in_stage = defaultdict(list)
    budgets_in_stage = defaultdict(list)

    for s in range(s_max + 1):
        n_configs_in_stage[s], budgets_in_stage[s] = compute_successive_halving(
            b_min,
            b_max,
            eta,
            early_stopping_rate=s
        )

    return s_max, n_configs_in_stage, budgets_in_stage

def compute_hyperband_budgets(
        b_min: float,
        b_max: float,
        eta: int,
        *,
        n_stages: int | None = None,
        print_output: bool = True,
        sample_default_at_target: bool = False,
    ) -> tuple[dict[int, list], dict[int, list], dict[int, list], int, float, float]:
    """Compute the number of configurations and budgets for each stage in HyperBand.

    Parameters
    ----------
    b_min : float
        Minimum budget.

    b_max : float
        Maximum budget.

    eta : int
        Reduction factor.

    n_stages : int, optional
        Number of stages to compute. Defaults to None. If None, all stages are computed.

    print_output : bool, optional
        Whether to print the output. Defaults to True.

    sample_default_at_target : bool, optional
        Whether to sample the default configuration at the target budget.
        Defaults to False.

    Returns:
    -------
    tuple[dict[int, list], dict[int, list], dict[int, list], int, float, float]
        Number of configurations, budgets, real budgets, total number of trials,
        total budget, total real budget.
    """
    s_max, n_configs_in_stage, budgets_in_stage = compute_hyperband_brackets(
        b_min,
        b_max,
        eta
    )

    total_real_budget = 0
    total_budget = 0

    total_trials = 0
    total_configs = 0

    epochs_in_stage = {}

    # Real budgets using checkpointing
    real_epochs_in_stage = {}
    for i in range(len(budgets_in_stage)):
        epochs_in_stage[i] = [round(b) for b in budgets_in_stage[i]]

        # First budgets stays the same
        real_epochs_in_stage[i] = [epochs_in_stage[i][0]]

        for epochs, prev_epochs in zip(
            epochs_in_stage[i][:-1],
            epochs_in_stage[i][1:],
            strict=False
        ):
            # We only train the difference
            real_epochs_in_stage[i].append(prev_epochs - epochs)

    # Now select how many stages we want
    if n_stages is None:
        n_stages = s_max + 1

    if sample_default_at_target:
        # We start by evaluationg the default configuration
        n_configs_in_stage[0] = [1] + n_configs_in_stage[0]
        budgets_in_stage[0] = [b_max] + budgets_in_stage[0]
        real_epochs_in_stage[0] = [b_max] + real_epochs_in_stage[0]

    for i in range(n_stages):
        configs_list = "[" + ", ".join(
            [str(c).rjust(8) for c in n_configs_in_stage[i]]) + "]"
        budgets_list = "[" + ", ".join(
            [str(round(b, 2)).rjust(8) for b in budgets_in_stage[i]]) + "]"
        epochs_list = "[" + ", ".join(
            [str(round(b, 2)).rjust(8) for b in epochs_in_stage[i]]) + "]"
        real_epochs_list = "[" + ", ".join(
            [str(round(b, 2)).rjust(8) for b in real_epochs_in_stage[i]]) + "]"

        stage_budget = sum([n * b for n, b in zip(
            n_configs_in_stage[i], budgets_in_stage[i], strict=False)])
        stage_real_budget = sum([n * b for n, b in zip(
            n_configs_in_stage[i], real_epochs_in_stage[i], strict=False)])

        total_trials += sum(n_configs_in_stage[i])

        # In hyperband, only the first budget in each stages
        # contains new configurations. In PriorBand, we have to
        # check whether this is the default configuration (i=0)
        if sample_default_at_target and i == 0:
            total_configs += n_configs_in_stage[i][1]
        else:
            total_configs += n_configs_in_stage[i][0]

        total_budget += stage_budget
        total_real_budget += stage_real_budget

        if print_output:
            print("-" * 80)
            print(f"Stage {i}: Budget {round(stage_budget, 2)}, "\
                  f"Real Budget {round(stage_real_budget, 2)}")
            print(f"  #Configs:     {configs_list}")
            print(f"  Budgets:      {budgets_list}")
            print(f"  Epochs:       {epochs_list}")
            print(f"  Real Epochs:  {real_epochs_list}")
            print("-" * 80)

    if print_output:
        print("Number of HB brackets: ", s_max + 1)
        print("Total number of trials: ", total_trials)
        print("Total number of configs: ", total_configs)
        print("Total budget: ", total_budget)
        print("Total real budget: ", total_real_budget)

    return (
        n_configs_in_stage,
        budgets_in_stage,
        real_epochs_in_stage,
        total_trials,
        total_budget,
        total_real_budget
    )

def get_budget_per_config(
        n_configs_in_stage: dict[int, list],
        budgets_in_stage: dict[int, list],
    ) -> dict[int, float]:
    """Returns a mapping from config IDs to budgest in the given HyperBand brackets.

    Parameters
    ----------
    n_configs_in_stage : dict[int, list]
        Number of configurations in each stage.

    budgets_in_stage : dict[int, list]
        Budgets in each stage.

    Returns:
    --------
    dict[int, float]
        Mapping from config IDs to budgets.
    """
    budget_per_config = {}

    config_id = 0
    for stage, n_configs in n_configs_in_stage.items():
        for n_config, real_budget in zip(
            n_configs,
            budgets_in_stage[stage],
            strict=False
        ):
            for _ in range(n_config):
                budget_per_config[config_id] = real_budget
                config_id += 1

    return budget_per_config