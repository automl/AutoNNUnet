from ConfigSpace import (
    ConfigurationSpace,
    Configuration,
    UniformFloatHyperparameter as Float,
    UniformIntegerHyperparameter as Integer,
    CategoricalHyperparameter as Categorical,
    EqualsCondition,
)
from pathlib import Path
import yaml
import pandas as pd
from deepcave.runs.recorder import Recorder
from deepcave.runs.objective import Objective
from deepcave.runs.converters.deepcave import DeepCAVERun
import numpy as np

def format_hp_name(name: str) -> str:
    return name.split(".")[-1]

def yaml_to_configspace(yaml_path: Path) -> ConfigurationSpace:
    with open(yaml_path, "r") as file:
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


def row_to_config(row: pd.Series, config_space: ConfigurationSpace) -> Configuration:
    values = {}

    for name, value in row.items():
        name = format_hp_name(str(name))
        if name in list(config_space.keys()):
            values[name] = value

    if values["optimizer"] != "SGD":
        values.pop("momentum")

    return Configuration(
        configuration_space=config_space,
        values=values,
        config_id=int(row["Run ID"]),
    )

def data_to_deepcave(dataset: str, history: pd.DataFrame, approach: str) -> DeepCAVERun:
    save_path = Path(f"./output/deepcave_logs").resolve()
    prefix = f"{dataset}_{approach}"

    if (save_path / prefix).exists():
        return DeepCAVERun.from_path(save_path / prefix)

    config_space_path = Path(f"./runscripts/configs/search_space/{approach}.yaml").resolve()
    config_space = yaml_to_configspace(config_space_path)

    for col in history.columns:
        if not "hp_config" in col:
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

    dice_objective = Objective("1 - Dice", lower=0, upper=1, optimize="lower")

    with Recorder(
        configspace=config_space,
        objectives=[dice_objective],
        save_path=str(save_path),
        prefix=prefix,
        overwrite=True
    ) as r:
        for _, run in history.iterrows():
            config = row_to_config(run, config_space)

            for fold in range(5):
                r.start(config=config, budget=run["Budget"], seed=fold)
                r.end(config=config, costs=run[f"o0_loss_fold_{fold}"], budget=run["Budget"], seed=fold)

    return DeepCAVERun.from_path(save_path / prefix)

