from __future__ import annotations

import argparse

from autonnunet.utils import compute_hyperband_budgets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--b_min", type=int, default=10)
    parser.add_argument("--b_max", type=int, default=1000)
    parser.add_argument("--eta", type=int, default=3)
    parser.add_argument("--sample_default_at_target", required=False, type=bool, default=False)
    args = parser.parse_args()

    compute_hyperband_budgets(
        b_min=args.b_min,
        b_max=args.b_max,
        eta=args.eta,
        sample_default_at_target=args.sample_default_at_target
    )