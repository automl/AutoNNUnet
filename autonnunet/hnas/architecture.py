from __future__ import annotations

import neps


def get_architecture(n_stages: int) -> neps.CFGArchitectureParameter: # type: ignore
    primitives = [
        "unet",
        "conv_encoder",
        "res_encoder",
        "down"
        "conv_decoder",
        "res_decoder",
        "up",
        "block",
        "instance_norm",
        "batch_norm",
        "leaky_relu",
        "relu",
        "prelu",
        "gelu",
        "1b",
        "2b",
        "3b",
        "4b",
        "5b",
        "6b",
    ]

    def get_part(part: str, _n_stages: range) -> list[str]:
        if "encoder" in part:
            return [f"{part}(NORM, NONLIN, DROPOUT, {', '.join(['B, down'] * n)}, B)" for n in _n_stages[:-1]]
        elif "decoder" in part:
            return [f"{part}(NORM, NONLIN, DROPOUT, {', '.join(['up, B'] * n)})" for n in _n_stages[:-1]]
        else:
            raise ValueError(f"Unknown part: {part}")
        
    # We want to keep at least half of the stages to ensure that the network is deep enough
    possible_n_stages = range(n_stages // 2, n_stages + 1)
    encoders =  get_part("conv_encoder", possible_n_stages) + get_part("res_encoder", possible_n_stages)
    decoders = get_part("conv_decoder", possible_n_stages) + get_part("res_decoder", possible_n_stages)

    structure = {
        "S": ["unet(E D)"],
        "E": encoders,
        "D": decoders,
        "B": ["1b", "2b", "3b", "4b", "5b", "6b"],
        "NORM": ["instance_norm", "batch_norm"],
        "NONLIN": ["leaky_relu", "relu", "prelu", "gelu"],
        "DROPOUT": ["dropout", "none"],
    }

    return neps.CFGArchitectureParameter(
        structure=structure,
        primitives=primitives,
    )


def get_default_architecture(n_stages: int) -> str:
    encoder_blocks_and_stages = ', '.join([f"b2, down" for _ in range(n_stages - 1)] + ["b2"])
    decoder_blocks_and_stages = ', '.join([f"up, b2" for _ in range(n_stages - 1)])

    return f"unet(conv_encoder(instance_norm, leaky_relu, no_dropout, {encoder_blocks_and_stages})"\
           f", conv_decoder(instance_norm, leaky_relu, no_dropout, {decoder_blocks_and_stages}))"


if __name__ == "__main__":
    default_architecture = get_default_architecture(n_stages=4)
    print(default_architecture)



