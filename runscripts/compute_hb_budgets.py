"""Compute the budgets for Hyperband."""
from automis.utils import compute_hyperband_budgets
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--b_min", type=int, default=15)
    parser.add_argument("--b_max", type=int, default=1000)
    parser.add_argument("--eta", type=int, default=4)
    args = parser.parse_args()

    compute_hyperband_budgets(
        b_min=args.b_min,
        b_max=args.b_max,
        eta=args.eta
    )