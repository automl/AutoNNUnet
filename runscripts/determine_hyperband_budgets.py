from smac.intensifier.hyperband_utils import determine_hyperband_for_multifidelity, print_hyperband_summary
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_budget", type=int, default=15000)
    parser.add_argument("--min_budget", type=int, default=100)
    parser.add_argument("--max_budget", type=int, default=1000)
    parser.add_argument("--eta", type=float, default=4)
    args = parser.parse_args()

    hyperband_info = determine_hyperband_for_multifidelity(
        total_budget=args.total_budget,
        min_budget=args.min_budget,
        max_budget=args.max_budget,
        eta=args.eta
    )

    print_hyperband_summary(hyperband_info)
