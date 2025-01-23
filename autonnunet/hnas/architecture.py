from __future__ import annotations
from typing import Literal
import numpy as np
from scipy.stats import truncnorm
import neps
from autonnunet.hnas.utils import plot_norm_dist, plot_cat_dist
from neps.search_spaces.hyperparameters import CategoricalParameter, IntegerParameter

PRIMITIVES = [
    "unet",
    "conv_encoder",
    "res_encoder",
    "down",
    "conv_decoder",
    "res_decoder",
    "up",
    "block",
    "instance_norm",
    "batch_norm",
    "leaky_relu",
    "relu",
    "elu",
    "prelu",
    "gelu",
    "1b",
    "2b",
    "3b",
    "4b",
    "5b",
    "6b",
    "7b",
    "8b",
    "9b",
    "10b",
    "11b",
    "12b",
    "dropout",
    "no_dropout",
]

# nnunetv2.experiment_planning.experiment_planners.default_experiment_planner.py#L61
CONV_BLOCKS_PER_STAGE_ENCODER = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
CONV_BLOCKS_PER_STAGE_DECODER = (2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)

# nnunetv2.experiment_planning.experiment_planners.resencUNet_planner.py#L27
RES_BLOCKS_PER_STAGE_ENCODER = (1, 3, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6)


def get_truncated_normal(lower: int, upper: int, default: int, confidence: str, title: str | None = None, plot: bool = False) -> np.ndarray:
    # We do this to assign equal probabilities to all buckets,
    # see neps.search_spaces.hyperparameters.integer.py#L83
    adjusted_lower = lower - 0.499999
    adjusted_upper = upper + 0.499999

    std = (upper - lower) * IntegerParameter.DEFAULT_CONFIDENCE_SCORES[confidence]
    a, b = (adjusted_lower - default) / std, (adjusted_upper - default) / std

    bucket_size = 1

    x = np.linspace(adjusted_lower, adjusted_upper, 10000 * (upper - lower))
    pdf = truncnorm.pdf(x, a, b, loc=default, scale=std)

    bucket_edges = np.arange(lower - (bucket_size / 2), upper + bucket_size, bucket_size)
    bucket_centers = (bucket_edges[:-1] + bucket_edges[1:]) / 2

    # We approximate the integral/probability of each bucket by summing the pdf values in the bucket,
    # and multiplying by the width of the bucket
    bucket_probs = np.zeros(len(bucket_centers))
    for i in range(len(bucket_edges) - 1):
        mask = (x >= bucket_edges[i]) & (x < bucket_edges[i + 1])
        bucket_probs[i] = np.sum(pdf[mask]) * (x[1] - x[0]) 

    bucket_probs /= bucket_probs.sum()

    if plot:        
        assert title is not None, "Title must be provided for plotting"
        plot_norm_dist(
            x=x,
            pdf=pdf,
            bucket_probs=bucket_probs,
            bucket_centers=bucket_centers,
            default=default,
            lower=lower,
            upper=upper,
            title=title
        )

    return bucket_probs


def get_cat_dist(
        num_categories: int,
        default: int,
        confidence: str,
        plot: bool = False,
        title: str | None = None
) -> np.ndarray:
    prior = np.ones(num_categories)
    prior[default] = CategoricalParameter.DEFAULT_CONFIDENCE_SCORES[confidence]

    prior /= np.sum(prior)

    if plot:
        assert title is not None, "Title must be provided for plotting"
        plot_cat_dist(
            prior=prior,
            default=default,
            title=title
        )

    return prior


def get_structure(
        n_stages: int, 
        s_max: int,
        prior_confidence: Literal["low", "medium", "high"],
        plot: bool = False
        ) -> tuple[dict, dict]:
    # We want to keep at least half of the stages to ensure that the network is deep enough
    possible_n_stages = range(n_stages // 2, n_stages + 1)
    starting_rule = [f"unet {n}E {n}D" for n in possible_n_stages]
    starting_rule_dist = get_truncated_normal(
        lower=n_stages // 2,
        upper=n_stages,
        default=n_stages,
        confidence=prior_confidence,
        title="S",
        plot=plot
    )

    def get_productions_and_prior(
            _n_stages: range
        ) -> tuple[dict[str, list[str]], dict[str, list[float]]]:
        result = {}
        prior = {}

        for n in _n_stages:
            result[f"{n}E"] = [
                f"conv_encoder ENORM ENONLIN EDROPOUT {' '.join([f'{_n}CEB, down' for _n in range(1, n)])}, {n}CEB",
                f"res_encoder ENORM ENONLIN EDROPOUT {' '.join([f'{_n}REB, down' for _n in range(1, n)])}, {n}REB"
            ]
            prior[f"{n}E"] = get_cat_dist(
                num_categories=2,
                default=0,
                confidence=prior_confidence,
                plot=plot,
                title=f"{n}E",
            )
            
            result[f"{n}D"] = [
                f"conv_decoder DNORM DNONLIN DDROPOUT {' '.join([f'up, {_n}DB' for _n in range(1, n)])}",
            ]
            prior[f"{n}D"] = get_cat_dist(
                num_categories=1,
                default=0,
                confidence=prior_confidence,
                plot=plot,
                title=f"{n}D",
            )

        for _n in range(1, max(_n_stages) + 1):
            conv_default = CONV_BLOCKS_PER_STAGE_ENCODER[_n - 1]
            res_default = RES_BLOCKS_PER_STAGE_ENCODER[_n - 1]

            result[f"{_n}CEB"] = [f"{i}b" for i in range(1, int(s_max *  conv_default) + 1)]
            result[f"{_n}REB"] = [f"{i}b" for i in range(1, int(s_max * res_default) + 1)]

            prior[f"{_n}CEB"] = get_truncated_normal(
                lower=1,
                upper=len(result[f"{_n}CEB"]),
                default=conv_default,
                confidence=prior_confidence,
                plot=plot,
                title=f"{_n}CEB"
            )
            prior[f"{_n}REB"] = get_truncated_normal(
                lower=1,
                upper=len(result[f"{_n}REB"]),
                default=res_default,
                confidence=prior_confidence,
                plot=plot,
                title=f"{_n}REB"
            )

        for _n in range(1, max(_n_stages) + 1):
            conv_default = CONV_BLOCKS_PER_STAGE_DECODER[_n - 1]
            result[f"{_n}DB"] = [f"{i}b" for i in range(1, s_max *  conv_default + 1)]
            prior[f"{_n}DB"] = get_truncated_normal(
                lower=1,
                upper=len(result[f"{_n}DB"]),
                default=conv_default,
                confidence=prior_confidence,
                plot=plot,
                title=f"{_n}DB"
            )

        return result, prior

    enc_dec_rules, enc_dec_prior = get_productions_and_prior(_n_stages=possible_n_stages) 

    structure = {
        "S": starting_rule,
        **enc_dec_rules,
        "ENORM": ["instance_norm", "batch_norm"],
        "DNORM": ["instance_norm", "batch_norm"],
        "ENONLIN": ["leaky_relu", "relu", "elu", "prelu", "gelu"],
        "DNONLIN": ["leaky_relu", "relu", "elu", "prelu", "gelu"],
        "EDROPOUT": ["dropout", "no_dropout"],
        "DDROPOUT": ["dropout", "no_dropout"],
    }

    prior_dist = {
        "S": starting_rule_dist,
        **enc_dec_prior,
        "ENORM": get_cat_dist(
            num_categories=len(structure["ENORM"]),
            default=0,
            confidence=prior_confidence,
            plot=plot,
            title="ENORM"
        ),
        "DNORM": get_cat_dist(
            num_categories=len(structure["DNORM"]),
            default=0,
            confidence=prior_confidence,
            plot=plot,
            title="DNORM"
        ),
        "ENONLIN": get_cat_dist(
            num_categories=len(structure["ENONLIN"]),
            default=0,
            confidence=prior_confidence,
            plot=plot,
            title="ENONLIN"
        ),
        "DNONLIN": get_cat_dist(
            num_categories=len(structure["DNONLIN"]),
            default=0,
            confidence=prior_confidence,
            plot=plot,
            title="DNONLIN"
        ),
        "EDROPOUT": get_cat_dist(
            num_categories=len(structure["EDROPOUT"]),
            default=1,
            confidence=prior_confidence,
            plot=plot,
            title="EDROPOUT"
        ),
        "DDROPOUT": get_cat_dist(
            num_categories=len(structure["DDROPOUT"]),
            default=1,
            confidence=prior_confidence,
            plot=plot,
            title="DDROPOUT"
        ),
    }

    return structure, prior_dist


def get_architecture(
        n_stages: int,
        s_max: int,
        prior_sampling_mode: Literal["mutation", "distribution"],
        prior_confidence: Literal["low", "medium", "high"]
    ) -> neps.CFGArchitectureParameter: # type: ignore
    structure, prior = get_structure(
        n_stages=n_stages,
        s_max=s_max,
        prior_confidence=prior_confidence,
    )

    return neps.CFGArchitectureParameter(
        structure=structure,
        primitives=PRIMITIVES,
        prior=prior,
        prior_sampling_mode=prior_sampling_mode
    )

def get_default_architecture(n_stages: int) -> str:
    encoder_blocks_and_stages = ' '.join([f'({_n}CEB 2b) down' for _n in range(1, n_stages)]) + f" ({n_stages}EB 2b)"
    decoder_blocks_and_stages = ' '.join([f'up ({_n}DB 2b)' for _n in range(1, n_stages)])

    arch = f"(S unet ({n_stages}E conv_encoder (ENORM instance_norm) (ENONLIN leaky_relu) (EDROPOUT no_dropout) {encoder_blocks_and_stages}) ({n_stages}D conv_decoder (DNORM instance_norm) (DNONLIN leaky_relu) (DDROPOUT no_dropout) {decoder_blocks_and_stages}))"

    return arch


def compute_search_space_size(structure: dict) -> int:
    def f(A: str) -> int:
        if A not in structure:  # Terminal case
            return 1
        return int(np.sum(
            [np.prod([f(A_prime) for A_prime in production.split(" ")])
            for production in structure[A]]
        ))

    return f("S")


if __name__ == "__main__":
    # structure, prior_dist = get_structure(n_stages=7, prior_confidence="medium")

    # for i in range(4, 8):
    #     print(f"Search Space Size for {i} stages:", compute_search_space_size(get_structure(n_stages=i, prior_confidence="medium")[0]))

    arch = get_architecture(n_stages=4, s_max=2, prior_sampling_mode="distribution", prior_confidence="medium")
    print(arch.sample().value)
    # default_arch = get_default_architecture(n_stages=4)
    # print(default_arch)

