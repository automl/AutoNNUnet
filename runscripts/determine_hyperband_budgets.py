from __future__ import annotations

import argparse

from autonnunet.utils import compute_hyperband_budgets
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_budget", type=int, default=10)
    parser.add_argument("--max_budget", type=int, default=1000)
    parser.add_argument("--eta", type=int, default=3)
    parser.add_argument("--n_stages", type=int, required=False)
    args = parser.parse_args()

    compute_hyperband_budgets(
        b_min=args.min_budget,
        b_max=args.max_budget,
        eta=args.eta,
        n_stages=args.n_stages,
        is_prior_band=True
    )