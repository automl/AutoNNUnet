from __future__ import annotations

import neps


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
    "dropout",
    "no_dropout",
]


def get_structure(n_stages: int) -> dict:
    # We want to keep at least half of the stages to ensure that the network is deep enough
    possible_n_stages = range(n_stages // 2, n_stages + 1)
    starting_rule = [f"unet {n}E {n}D" for n in possible_n_stages]

    def get_productions_dict(part: str, _n_stages: range) -> dict[str, list[str]]:
        result = {}
        for n in _n_stages:
            if part == "encoder":
                result[f"{n}E"] = [f"conv_{part} ENORM, ENONLIN, EDROPOUT, {', '.join([f'{_n}EB, down' for _n in range(1, n)])}, {n}EB"]
                result[f"{n}E"] = [f"res_{part} ENORM, ENONLIN, EDROPOUT, {', '.join([f'{_n}EB, down' for _n in range(1, n)])}, {n}EB"]

            elif part == "decoder":
                result[f"{n}D"] = [f"conv_{part} DNORM, DNONLIN, DDROPOUT, {', '.join([f'up, {_n}DB' for _n in range(1, n)])}"]
                result[f"{n}D"] = [f"res_{part} DNORM, DNONLIN, DDROPOUT, {', '.join([f'up, {_n}DB' for _n in range(1, n)])}"]

            else:
                raise ValueError(f"Unknown part: {part}")
            
        for _n in range(1, max(_n_stages) + 1):
            result[f"{_n}EB"] = ["1b", "2b", "3b", "4b", "5b", "6b"]

        for _n in range(1, max(_n_stages) + 1):
            result[f"{_n}DB"] = ["1b", "2b", "3b", "4b", "5b", "6b"]
        return result

    encoder_rules = get_productions_dict("encoder", possible_n_stages) 
    decoder_rules = get_productions_dict("decoder", possible_n_stages)

    return {
        "S": starting_rule,
        **encoder_rules,
        **decoder_rules,
        "ENORM": ["instance_norm", "batch_norm"],
        "ENONLIN": ["leaky_relu", "relu", "elu", "prelu", "gelu"],
        "EDROPOUT": ["dropout", "no_dropout"],
        "DNORM": ["instance_norm", "batch_norm"],
        "DNONLIN": ["leaky_relu", "relu", "elu", "prelu", "gelu"],
        "DDROPOUT": ["dropout", "no_dropout"],
    }


def get_architecture(n_stages: int) -> neps.CFGArchitectureParameter: # type: ignore
    structure = get_structure(n_stages=n_stages)

    return neps.CFGArchitectureParameter(
        structure=structure,
        primitives=PRIMITIVES,
    )


def get_default_architecture(n_stages: int) -> str:
    encoder_blocks_and_stages = ', '.join([f'({_n}EB 2b) down' for _n in range(1, n_stages)]) + f", ({n_stages}EB 2b)"
    decoder_blocks_and_stages = ', '.join([f'up ({_n}DB 2b)' for _n in range(1, n_stages)])

    arch = f"(S unet ({n_stages}E conv_encoder (ENORM instance_norm) (ENONLIN leaky_relu) (EDROPOUT no_dropout) {encoder_blocks_and_stages}) ({n_stages}D conv_decoder (DNORM instance_norm) (DNONLIN leaky_relu) (DDROPOUT no_dropout) {decoder_blocks_and_stages}))"

    return arch


if __name__ == "__main__":
    default_architecture = get_architecture(n_stages=4)
    print(default_architecture)

