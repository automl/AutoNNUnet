"""Utility functions for converting between HyperSweeper and DeepCAVE formats."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from autonnunet.hnas.cfg_unet import CFGUNet
from ConfigSpace import (
    CategoricalHyperparameter as Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    OrConjunction,
    UniformFloatHyperparameter as Float,
    UniformIntegerHyperparameter as Integer,
)
from deepcave.runs.converters.deepcave import DeepCAVERun
from deepcave.runs.objective import Objective
from deepcave.runs.recorder import Recorder

if TYPE_CHECKING:
    import pandas as pd

# nnunetv2.experiment_planning.experiment_planners.default_experiment_planner.py#L61
# nnunetv2.experiment_planning.experiment_planners.resencUNet_planner.py#L27

# For the encoder, we take the maximum of the two presets
MAX_BLOCKS_PER_STAGE_ENCODER =  np.array([2, 3, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]) * 2
# For the decoder, we use the convolutional preset since it has more blocks
MAX_BLOCKS_PER_STAGE_DECODER = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]) * 2

O_DSC = "1 - DSC [%]"
O_RUNTIME = "Runtime [h]"

def format_hp_name(name: str) -> str:
    """Formats the hyperparameter name be removing the prefix.

    Parameters
    ----------
    name : str
        The hyperparameter name.

    Returns:
    -------
    str
        The formatted hyperparameter name.
    """
    return name.split(".")[-1]

def yaml_to_configspace(yaml_path: Path) -> ConfigurationSpace:
    """Converts a yaml file to a ConfigSpace object.

    Parameters
    ----------
    yaml_path : Path
        The path to the yaml file.

    Returns:
    -------
    ConfigurationSpace
        The ConfigSpace object.

    Raises:
    ------
    ValueError
        If the hyperparameter type is unknown.
    """
    with open(yaml_path) as file:
        config_space_dict = yaml.safe_load(file)

    config_space = ConfigurationSpace()

    for name, hp in config_space_dict["hyperparameters"].items():
        name = format_hp_name(name)

        if hp["type"] == "uniform_float":
            config_space.add(
                Float(
                    name=name,
                    lower=hp["lower"],
                    upper=hp["upper"],
                    default_value=hp.get("default", None),
                    log=hp.get("log", False),
                )
            )
        elif hp["type"] == "uniform_int":
            config_space.add(
                Integer(
                    name=name,
                    lower=hp["lower"],
                    upper=hp["upper"],
                    default_value=hp.get("default", None),
                    log=hp.get("log", False),
                )
            )
        elif hp["type"] == "categorical":
            config_space.add(
                Categorical(
                    name=name,
                    choices=hp["choices"],
                    default_value=hp.get("default", None),
                )
            )
        else:
            raise ValueError(f"Unknown hyperparameter type: {hp['type']}")


    for cond in config_space_dict["conditions"]:
        if cond["type"] == "EQ":
            config_space.add(
                EqualsCondition(
                    child=config_space[format_hp_name(cond["child"])],
                    parent=config_space[format_hp_name(cond["parent"])],
                    value=cond["value"])
                )
        else:
            raise ValueError(f"Unknown condition type: {cond['type']}")

    return config_space

def extract_architecture_features(string_tree: str) -> dict:
    """Extracts the architecture features from a hierarchical architecture string.

    Parameters
    ----------
    string_tree : str
        The hierarchical architecture string.

    Returns:
    -------
    dict
        The architecture features.
    """
    string_tree = str(string_tree).replace('"', "")

    parsed_tree = CFGUNet.parse_nested_brackets(string_tree)
    encoder_cfg, decoder_cfg = CFGUNet.extract_architecture_cfg(parsed_tree)

    return {
        "n_stages": len(encoder_cfg["n_blocks_per_stage"]),

        "encoder_type": encoder_cfg["network_type"],
        "encoder_norm": encoder_cfg["norm"],
        "encoder_nonlin": encoder_cfg["nonlin"],
        "encoder_dropout": encoder_cfg["dropout"],
        "encoder_depth": sum(encoder_cfg["n_blocks_per_stage"][:-1]),

        "decoder_norm": decoder_cfg["norm"],
        "decoder_nonlin": decoder_cfg["nonlin"],
        "decoder_dropout": decoder_cfg["dropout"],
        "decoder_depth": sum(decoder_cfg["n_blocks_per_stage"]),

        "bottleneck_depth": encoder_cfg["n_blocks_per_stage"][-1],
    }


def row_to_config(row: pd.Series, config_space: ConfigurationSpace) -> Configuration:
    """Extracts a hyperparameter configuration from a row in the runhistory.

    Parameters
    ----------
    row : pd.Series
        The row in the runhistory.

    config_space : ConfigurationSpace
        The configuration space object.

    Returns:
    -------
    Configuration
        The hyperparameter configuration.
    """
    values = {}

    # We need to manually collect the hyperparameter configuraiton
    # from the corresponding row in the runhistory
    for name, value in row.items():
        name = format_hp_name(str(name))
        if name in list(config_space.keys()):
            values[name] = value

    # To match the config space, we need to pretend
    # that the learning rate was not sampled
    if values["optimizer"] != "SGD":
        values.pop("momentum")

    if "architecture" in row:
        # This means were in HPO + HNAS
        architecture_features = extract_architecture_features(row["architecture"])
        values.update(architecture_features)

        # To match the config space, we need to pretend that the dropout rate was not sampled
        if values["encoder_dropout"] != "dropout" and values["decoder_dropout"] != "dropout":
            values.pop("dropout_rate")

    origin = row.get("Config Origin", None)

    return Configuration(
        configuration_space=config_space,
        values=values,
        config_id=int(row["Configuration ID"]),
        origin=origin
    )

def get_extended_hnas_config_space(
        config_space: ConfigurationSpace,
        default_string_tree: str
    ) -> ConfigurationSpace:
    """Extends the given configuration space with the hierarchical architecture features.

    Parameters
    ----------
    config_space : ConfigurationSpace
        The configuration space object.

    default_string_tree : str
        The default hierarchical architecture string.

    Returns:
    -------
    ConfigurationSpace
        The extended configuration space object.
    """
    default_features = extract_architecture_features(default_string_tree)

    config_space.add([
        Integer(
            name="n_stages",
            lower=1,
            upper=default_features["n_stages"],
            default_value=default_features["n_stages"],
        ),
        Categorical(
            name="encoder_type",
            choices=["conv_encoder", "res_encoder"],
            default_value=default_features["encoder_type"],
        ),
        Categorical(
            name="encoder_norm",
            choices=["instance_norm", "batch_norm"],
            default_value=default_features["encoder_norm"],
        ),
        Categorical(
            name="encoder_nonlin",
            choices=["leaky_relu", "relu", "elu", "gelu", "prelu"],
            default_value="leaky_relu",
        ),
        Categorical(
            name="encoder_dropout",
            choices=["dropout", "no_dropout"],
            default_value=default_features["encoder_dropout"],
        ),
        Integer(
            name="encoder_depth",
            lower=default_features["n_stages"] // 2 - 1,
            upper=sum(MAX_BLOCKS_PER_STAGE_ENCODER[:default_features["n_stages"] - 1]),
            default_value=default_features["encoder_depth"],
        ),
        Categorical(
            name="decoder_norm",
            choices=["instance_norm", "batch_norm"],
            default_value=default_features["decoder_norm"],
        ),
        Categorical(
            name="decoder_nonlin",
            choices=["leaky_relu", "relu", "elu", "gelu", "prelu"],
            default_value="leaky_relu",
        ),
        Categorical(
            name="decoder_dropout",
            choices=["dropout", "no_dropout"],
            default_value=default_features["decoder_dropout"],
        ),
        Integer(
            name="decoder_depth",
            lower=default_features["n_stages"] // 2 - 1,
            upper=sum(MAX_BLOCKS_PER_STAGE_DECODER[:default_features["n_stages"] - 1]),
            default_value=default_features["decoder_depth"],
        ),
        Integer(
            name="bottleneck_depth",
            lower=1,
            upper=int(MAX_BLOCKS_PER_STAGE_ENCODER[default_features["n_stages"] - 1]),
            default_value=default_features["bottleneck_depth"],
        ),
    ])

    # We need to ensure that the dropout rate is only active if dropout is enabled
    config_space.add(
        OrConjunction(
            EqualsCondition(
                child=config_space["dropout_rate"],
                parent=config_space["encoder_dropout"],
                value="dropout"
            ),
            EqualsCondition(
                child=config_space["dropout_rate"],
                parent=config_space["decoder_dropout"],
                value="dropout"
            )
        ),
    )

    return config_space


def runhistory_to_deepcave(dataset: str, history: pd.DataFrame, approach_key: str) -> DeepCAVERun:
    """Converts a hypersweeper runhistory to a DeepCAVE run.

    Parameters
    ----------
    dataset : str
        The dataset name.

    history : pd.DataFrame
        The hypersweeper runhistory.

    approach_key : str
        The approach key (hpo, hpo_nas, hpo_hnas).

    Returns:
    -------
    DeepCAVERun
        The DeepCAVE run.

    Raises:
    ------
    ValueError
        If the hyperparameter type is unknown.
    """
    save_path = Path("./output/deepcave_logs").resolve()
    prefix = f"{dataset}_{approach_key}"

    if (save_path / prefix).exists():
        return DeepCAVERun.from_path(save_path / prefix)

    config_space_path = Path(f"./runscripts/configs/search_space/{approach_key}.yaml").resolve()
    config_space = yaml_to_configspace(config_space_path)

    # In HNAS, we need to extend the config space with the architecture features
    if "architecture" in history.columns:
        default_string_tree = str(history["architecture"].values[0])
        config_space = get_extended_hnas_config_space(
            config_space=config_space,
            default_string_tree=default_string_tree
        )

    for col in history.columns:
        if "hp_config" not in col:
            continue

        name = format_hp_name(col)

        if isinstance(config_space[name], Float):
            history[col] = history[col].astype(float)
        elif isinstance(config_space[name], Integer):
            history[col] = history[col].astype(int)
        elif isinstance(config_space[name], Categorical):
            history[col] = history[col].astype(str)
            history[col] = history[col].apply(lambda x: x.replace("nan", "None"))
        else:
            raise ValueError(f"Unknown hyperparameter type: {config_space[name]}")

    dice_objective = Objective(O_DSC, lower=0, upper=100, optimize="lower")
    if O_RUNTIME in history.columns:
        runtime_objective = Objective(O_RUNTIME, lower=0, optimize="lower")
        objectives = [dice_objective, runtime_objective]
    else:
        objectives = [dice_objective]

    with Recorder(
        configspace=config_space,
        objectives=objectives,
        save_path=str(save_path),
        prefix=prefix,
        overwrite=True
    ) as r:
        for _, run in history.iterrows():
            config = row_to_config(run, config_space)
            r.start(config=config, budget=run["Budget"], origin=config.origin)
            costs = [run[O_DSC]]
            if len(objectives) > 1:
                # For DeepCAVE, we need to normalize to the maximum budget
                # to ensure that the runtime is comparable across different budgets
                runtime = run[O_RUNTIME] / int(run["Budget"]) * 1000
                costs.append(runtime)
            r.end(config=config, costs=costs, budget=run["Budget"])      # type: ignore

    return DeepCAVERun.from_path(save_path / prefix)

