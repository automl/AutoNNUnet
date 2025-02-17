from __future__ import annotations

import argparse

from autonnunet.utils import compute_hyperband_budgets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_budget", type=int, default=10)
    parser.add_argument("--max_budget", type=int, default=1000)
    parser.add_argument("--eta", type=int, default=3)
    parser.add_argument("--sample_default_at_target", required=False, type=bool, default=False)
    args = parser.parse_args()

    compute_hyperband_budgets(
        b_min=args.min_budget,
        b_max=args.max_budget,
        eta=args.eta,
        sample_default_at_target=args.sample_default_at_target
    )