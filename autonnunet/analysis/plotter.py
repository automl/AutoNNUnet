from __future__ import annotations

import ast
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Any

import matplotlib.axis
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from deepcave.evaluators.fanova import fANOVA
from deepcave.evaluators.lpi import LPI
from deepcave.evaluators.mo_fanova import MOfANOVA
from deepcave.evaluators.mo_lpi import MOLPI
from deepcave.evaluators.footprint import Footprint
from deepcave.constants import COMBINED_BUDGET
from deepcave.plugins.budget.budget_correlation import BudgetCorrelation
from deepcave.evaluators.ablation import Ablation
from deepcave.evaluators.mo_ablation import MOAblation
from deepcave.plugins.hyperparameter.pdp import PartialDependencies
from deepcave.utils.styled_plotty import get_hyperparameter_ticks

from autonnunet.analysis.deepcave_utils import runhistory_to_deepcave
from autonnunet.utils import (compute_hyperband_budgets,
                              dataset_name_to_msd_task, format_dataset_name,
                              get_budget_per_config, load_json)
from autonnunet.utils.helpers import msd_task_to_dataset_name
from autonnunet.utils.paths import (AUTONNUNET_MSD_RESULTS, AUTONNUNET_OUTPUT,
                                    AUTONNUNET_PLOTS, AUTONNUNET_TABLES,
                                    NNUNET_DATASETS)

if TYPE_CHECKING:
    from deepcave.runs.converters.deepcave import DeepCAVERun

APPROACHES = [
    "hpo",
    "hpo_nas",
    "hpo_hnas"
]

APPROACH_REPLACE_MAP = {
    "baseline_ConvolutionalEncoder": "nnU-Net (Conv)",
    "baseline_ResidualEncoderM": "nnU-Net (ResM)",
    "baseline_ResidualEncoderL": "nnU-Net (ResL)",
    "hpo": "HPO (ours)",
    "hpo_nas": "HPO + NAS (ours)",
    "hpo_hnas": "HPO + HNAS (ours)"
}

PROGRESS_REPLACEMENT_MAP = {
    "mean_fg_dice": "Mean Foreground Dice",
    "ema_fg_dice": "EMA Foreground Dice",
    "train_losses": "Training Loss",
    "val_losses": "Validation Loss",
    "lrs": "Learning Rate",
}

HPO_HYPERPARAMETERS = [
    "optimizer",
    "momentum",
    "initial_lr",
    "lr_scheduler",
    "weight_decay",
    "loss_function",
    "aug_factor",
    "oversample_foreground_percent",
]

NAS_HYPERPARAMETERS = [
    "encoder_type", "Encoder Type",
    "model_scale", "Model Scale",
    "base_num_features", "Base #Features",
    "max_features", "Max. #Features",
    "activation", "Activation",
    "normalization", "Normalization",
    "dropout_rate", "Dropout Rate",
    "n_stages", "Stages",
]

HNAS_HYPERPARAMETERS = [
    "encoder_type", "Encoder Type",
    "base_num_features", "Base #Features",
    "max_features", "Max. #Features",
    "dropout_rate", "Dropout Rate",
    "n_stages", "Stages",
    "encoder_norm", "Encoder Norm.",
    "encoder_nonlin", "Encoder Activation",
    "encoder_dropout", "Encoder Dropout",
    "encoder_depth", "Encoder Depth",
    "decoder_norm", "Decoder Norm.",
    "decoder_nonlin", "Decoder Activation",
    "decoder_dropout", "Decoder Dropout",
    "decoder_depth", "Decoder Depth",
    "bottleneck_depth", "Bottleneck Depth",
]

HYPERPARAMETER_REPLACEMENT_MAP = {
    "default": "Default",
    "optimizer": "Optimizer",
    "momentum": "Momentum (SGD)",
    "initial_lr": "Initial LR",
    "lr_scheduler": "LR Scheduler",
    "weight_decay": "Weight Decay",
    "loss_function": "Loss Function",
    "aug_factor": "Data Aug. Factor",
    "oversample_foreground_percent": "Foreground Oversamp.",
    "encoder_type": "Encoder Type",
    "model_scale": "Model Scale",
    "base_num_features": "Base #Features",
    "max_features": "Max. #Features",
    "activation": "Activation",
    "normalization": "Normalization",
    "dropout_rate": "Dropout Rate",
    "n_stages": "Stages",
    "encoder_norm": "Encoder Norm.",
    "encoder_nonlin": "Encoder Activation",
    "encoder_dropout": "Encoder Dropout",
    "encoder_depth": "Encoder Depth",
    "decoder_norm": "Decoder Norm.",
    "decoder_nonlin": "Decoder Activation",
    "decoder_dropout": "Decoder Dropout",
    "decoder_depth": "Decoder Depth",
    "bottleneck_depth": "Bottleneck Depth",
}

HISTORY_REPLACEMENT_MAP = {
    "config_id": "Configuration ID",
    "run_id": "Run ID",
    "budget": "Budget",
    "o0_loss": "1 - Dice",
    "o1_runtime": "Runtime",
}

STYLES_TYPE = Literal["white", "dark", "whitegrid", "darkgrid", "ticks"]

OBJECTIVES_MAPPING = {
    "1 - Dice": "loss",
    "Runtime": "runtime"
}

ORIGIN_MAP = {
    "random": "Rand. Sampling",
    "prior": "Prior Sampling",
    "incumbent": "Inc. Sampling",
}

PROGRESS_FILENAME = "progress.csv"
HISTORY_FILENAME = "runhistory.csv"
SAMPLING_POLICY_LOGS = "sampling_policy.log"
INCUMBENT_FILENAME = "incumbent_loss.csv"
VALIDATION_METRICS_FILENAME = "summary.json"
EMISSIONS_FILENAME = "emissions.csv"
DATASET_JSON_FILENAME = "dataset.json"

DEEPCAVE_CACHE_DIR = Path("./output/deepcave_cache").resolve()
DEEPCAVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


WONG_PALETTE = [
    "#2271B2",
    "#E69F00",
    "#2C9E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
]

# print full dataframes
pd.set_option("display.max_rows", None)

@dataclass
class BaselineResult:
    progress: pd.DataFrame
    metrics_foreground_mean: pd.DataFrame
    metrics_mean: pd.DataFrame
    metrics_per_case: pd.DataFrame
    emissions: pd.DataFrame

@dataclass
class HPOResult:
    history: pd.DataFrame
    incumbent: pd.DataFrame
    incumbent_progress: pd.DataFrame
    emissions: pd.DataFrame
    deepcave_runs: dict[str, DeepCAVERun]

@dataclass
class NASResult:
    history: pd.DataFrame
    incumbents: dict[str, pd.DataFrame]
    emissions: pd.DataFrame
    deepcave_runs: dict[str, DeepCAVERun]

class Plotter:
    def __init__(
            self,
            datasets: list[str],
            configuration: str = "3d_fullres",
            objectives: list[str] | None = None,
            min_budget: float = 10.0,
            max_budget: float = 1000.0,
            eta: int = 3,
            n_folds: int = 5,
            style: STYLES_TYPE = "whitegrid",
            color_palette: list = sns.color_palette("colorblind"),
            figwidth: int = 8,
            hpo_seed: int = 0,
            dpi: int = 400,
            format: str = "png",
            lazy_loading: bool = True
        ):
        self.logger = logging.getLogger("Plotter")

        if objectives is None:
            objectives = ["1 - Dice", "Runtime"]
        self.datasets = datasets
        self.objectives = objectives

        # We need these to find the corresponding directories
        self.configuration = configuration
        self.n_folds = n_folds
        self.hpo_seed = hpo_seed

        self._setup_hyperband(min_budget=min_budget, max_budget=max_budget, eta=eta)
        self._setup_paths()

        # Seaborn settings
        sns.set_style(style=style)
        self.figwidth = figwidth
        self.dpi = dpi
        self.format = format
        self.color_palette = color_palette

        self._init_data()
        if not lazy_loading:
            self.load_all_data()

    def _init_data(self):
        self._baseline_data = BaselineResult(
            progress=pd.DataFrame(),
            metrics_foreground_mean=pd.DataFrame(),
            metrics_mean=pd.DataFrame(),
            metrics_per_case=pd.DataFrame(),
            emissions=pd.DataFrame()
        )
        self._baseline_datasets = []

        self._hpo_data = HPOResult(
            history=pd.DataFrame(),
            incumbent=pd.DataFrame(),
            incumbent_progress=pd.DataFrame(),
            emissions=pd.DataFrame(),
            deepcave_runs={}
        )
        self._hpo_datasets = []

        self._nas_data = NASResult(
            history=pd.DataFrame(),
            incumbents={k: pd.DataFrame() for k in self.objectives},
            emissions=pd.DataFrame(),
            deepcave_runs={}
        )
        self._nas_datasets = []

        self._hnas_data = NASResult(
            history=pd.DataFrame(),
            incumbents={k: pd.DataFrame() for k in self.objectives},
            emissions=pd.DataFrame(),
            deepcave_runs={}
        )
        self._hnas_datasets = []

    def _setup_hyperband(self, min_budget: float, max_budget: float, eta: int):
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.eta = eta
        
        # Hyperband configuration for HPO, here we start by sampling the
        # the default configuration at full budget
        (
            self.n_configs_in_stage,
            self.budgets_in_stage,
            self.real_budgets_in_stage,
            _,
            _,
            self.total_real_budget
        ) = compute_hyperband_budgets(
            b_min=min_budget,
            b_max=max_budget,
            eta=eta,
            print_output=False,
            sample_default_at_target=True
        )
        self.n_full_trainings = self.total_real_budget / self.max_budget
        self.show_n_full_trainings = 20

        # We need this to assign the real budget for each configuration
        self.real_budgets_per_config = get_budget_per_config(
            n_configs_in_stage=self.n_configs_in_stage,
            budgets_in_stage=self.real_budgets_in_stage
        )
    
    def _setup_paths(self):
        self.baseline_conv =  AUTONNUNET_OUTPUT / "baseline_ConvolutionalEncoder"
        self.baseline_resenc_m =  AUTONNUNET_OUTPUT / "baseline_ResidualEncoderM"
        self.baseline_resenc_l =  AUTONNUNET_OUTPUT / "baseline_ResidualEncoderL"
        self.hpo_dir = AUTONNUNET_OUTPUT / "hpo"
        self.nas_dir = AUTONNUNET_OUTPUT / "hpo_nas"
        self.hnas_dir = AUTONNUNET_OUTPUT / "hpo_hnas"

        self.hpo_plots = AUTONNUNET_PLOTS / "hpo"
        self.hpo_plots.mkdir(parents=True, exist_ok=True)
        self.nas_plots = AUTONNUNET_PLOTS / "hpo_nas"
        self.nas_plots.mkdir(parents=True, exist_ok=True)
        self.hnas_plots = AUTONNUNET_PLOTS / "hpo_hnas"
        self.hnas_plots.mkdir(parents=True, exist_ok=True)
        self.baseline_plots = AUTONNUNET_PLOTS / "baseline"
        self.baseline_plots.mkdir(parents=True, exist_ok=True)

        self.hpo_analysis_plots = AUTONNUNET_PLOTS / "hpo_analysis"
        self.hpo_analysis_plots.mkdir(parents=True, exist_ok=True)
        self.nas_analysis_plots = AUTONNUNET_PLOTS / "hpo_nas_analysis"
        self.nas_analysis_plots.mkdir(parents=True, exist_ok=True)
        self.hnas_analysis_plots = AUTONNUNET_PLOTS / "hpo_hnas_analysis"
        self.hnas_analysis_plots.mkdir(parents=True, exist_ok=True)

        self.analysis_plots = {
            "hpo": self.hpo_analysis_plots,
            "hpo_nas": self.nas_analysis_plots,
            "hpo_hnas": self.hnas_analysis_plots
        }

    def _load_baseline_data_lazy(self, dataset: str):
        if dataset in self._baseline_datasets:
            return

        baseline_data = self._load_baseline_data(datasets=[dataset])
        self._baseline_data.progress = pd.concat([self._baseline_data.progress, baseline_data.progress])
        self._baseline_data.emissions = pd.concat([self._baseline_data.emissions, baseline_data.emissions])
        self._baseline_data.metrics_foreground_mean = pd.concat([self._baseline_data.metrics_foreground_mean, baseline_data.metrics_foreground_mean])
        self._baseline_data.metrics_mean = pd.concat([self._baseline_data.metrics_mean, baseline_data.metrics_mean])
        self._baseline_data.metrics_per_case = pd.concat([self._baseline_data.metrics_per_case, baseline_data.metrics_per_case])

    def _load_hpo_data_lazy(self, dataset: str):
        if dataset in self._hpo_datasets:
            return
        
        hpo_data = self._load_hpo_data(datasets=[dataset])
        self._hpo_data.history = pd.concat([self._hpo_data.history, hpo_data.history])
        self._hpo_data.incumbent = pd.concat([self._hpo_data.incumbent, hpo_data.incumbent])
        self._hpo_data.incumbent_progress = pd.concat([self._hpo_data.incumbent_progress, hpo_data.incumbent_progress])
        self._hpo_data.emissions = pd.concat([self._hpo_data.emissions, hpo_data.emissions])
        self._hpo_data.deepcave_runs.update(hpo_data.deepcave_runs)

    def _load_nas_data_lazy(self, dataset: str, approach_key: str):
        if approach_key == "hpo_nas":
            data = self._nas_data
            datasets = self._nas_datasets
        elif approach_key == "hpo_hnas":
            data = self._hnas_data
            datasets = self._hnas_datasets
        else:
            raise ValueError(f"Unknown approach key {approach_key}.")

        if dataset in datasets:
            return

        nas_data = self._load_nas_data(datasets=[dataset], approach_key=approach_key)
        data.history = pd.concat([data.history, nas_data.history])
        data.incumbents = {k: pd.concat([v, nas_data.incumbents[k]]) for k, v in data.incumbents.items()}
        data.emissions = pd.concat([data.emissions, nas_data.emissions])
        data.deepcave_runs.update(nas_data.deepcave_runs)

    def load_all_data(self):
        self._baseline_data = self._load_baseline_data(datasets=self.datasets)
        self._baseline_datasets = self._baseline_data.progress["Dataset"].unique().tolist()
        self.logger.info(
            f"Loaded {len(self._baseline_datasets)} datasets for baseline.")   
       
        self._hpo_data = self._load_hpo_data(datasets=self.datasets)
        self._hpo_datasets = self._hpo_data.history["Dataset"].unique().tolist()
        self.logger.info(
            f"Loaded {len(self._hpo_datasets)} datasets for HPO.")
        
        self._nas_data = self._load_nas_data(datasets=self.datasets, approach_key="hpo_nas")
        self._nas_datasets = self._nas_data.history["Dataset"].unique().tolist()
        self.logger.info(
            f"Loaded {len(self._nas_datasets)} datasets for HPO + NAS.")
        
        self._hnas_data = self._load_nas_data(datasets=self.datasets, approach_key="hpo_hnas")
        self._hnas_datasets = self._nas_data.history["Dataset"].unique().tolist()
        self.logger.info(
            f"Loaded {len(self._hnas_datasets)} datasets for HPO + HNAS.")

    def get_baseline_data(self, dataset: str):
        if dataset not in self._baseline_datasets:
            self._load_baseline_data_lazy(dataset=dataset)

        progress = self._baseline_data.progress[
            self._baseline_data.progress["Dataset"] == dataset]
        emissions = self._baseline_data.emissions[
            self._baseline_data.emissions["Dataset"] == dataset]
        metrics_foreground_mean = self._baseline_data.metrics_foreground_mean[
            self._baseline_data.metrics_foreground_mean["Dataset"] == dataset]
        metrics_mean = self._baseline_data.metrics_mean[
            self._baseline_data.metrics_mean["Dataset"] == dataset]
        metrics_per_case = self._baseline_data.metrics_per_case[
            self._baseline_data.metrics_per_case["Dataset"] == dataset]

        return BaselineResult(
            progress=progress,
            emissions=emissions,
            metrics_foreground_mean=metrics_foreground_mean,
            metrics_mean=metrics_mean,
            metrics_per_case=metrics_per_case
        )

    def get_hpo_data(self, dataset: str):
        if dataset not in self._hpo_datasets:
            self._load_hpo_data_lazy(dataset=dataset)

        incumbent_progress = self._hpo_data.incumbent_progress[
            self._hpo_data.incumbent_progress["Dataset"] == dataset]
        emissions = self._hpo_data.emissions[
            self._hpo_data.emissions["Dataset"] == dataset]
        history = self._hpo_data.history[
            self._hpo_data.history["Dataset"] == dataset]
        incumbent = self._hpo_data.incumbent[
            self._hpo_data.incumbent["Dataset"] == dataset]

        return HPOResult(
            incumbent_progress=incumbent_progress,
            emissions=emissions,
            history=history,
            incumbent=incumbent,
            deepcave_runs={dataset: self._hpo_data.deepcave_runs[dataset]}
        )

    def get_nas_data(self, dataset: str):
        if dataset not in self._nas_datasets:
            self._load_nas_data_lazy(dataset=dataset, approach_key="hpo_nas")

        emissions = self._nas_data.emissions[
            self._nas_data.emissions["Dataset"] == dataset]
        history = self._nas_data.history[
            self._nas_data.history["Dataset"] == dataset]
        incumbents = {}
        for objective in self.objectives:
            incumbents[objective] = self._nas_data.incumbents[objective][
                self._nas_data.incumbents[objective]["Dataset"] == dataset]

        return NASResult(
            emissions=emissions,
            history=history,
            incumbents=incumbents,
            deepcave_runs={dataset: self._nas_data.deepcave_runs[dataset]}
        )
    
    def get_hnas_data(self, dataset: str):
        if dataset not in self._nas_datasets:
            self._load_nas_data_lazy(dataset=dataset, approach_key="hpo_hnas")
    
        emissions = self._hnas_data.emissions[
            self._hnas_data.emissions["Dataset"] == dataset]
        history = self._hnas_data.history[
            self._hnas_data.history["Dataset"] == dataset]
        incumbents = {}
        for objective in self.objectives:
            incumbents[objective] = self._hnas_data.incumbents[objective][
                self._hnas_data.incumbents[objective]["Dataset"] == dataset]

        return NASResult(
            emissions=emissions,
            history=history,
            incumbents=incumbents,
            deepcave_runs={dataset: self._hnas_data.deepcave_runs[dataset]}
        )

    def get_deepcave_data(self, dataset: str, approach_key: str, objective: str = "1 - Dice") -> tuple[DeepCAVERun, pd.DataFrame, pd.DataFrame]:
        if approach_key == "hpo":
            deepcave_run = self._hpo_data.deepcave_runs[dataset]
            history = self.get_hpo_data(dataset).history
            incumbent = self.get_hpo_data(dataset).incumbent
        elif approach_key == "hpo_nas":
            deepcave_run = self._nas_data.deepcave_runs[dataset]
            history = self.get_nas_data(dataset).history
            incumbent = self.get_nas_data(dataset).incumbents[objective]
        elif approach_key == "hpo_hnas":
            deepcave_run = self._hnas_data.deepcave_runs[dataset]
            history = self.get_hnas_data(dataset).history
            incumbent = self.get_hnas_data(dataset).incumbents[objective]
        else:
            raise ValueError(f"Unknown approach key {approach_key}.")
        
        return deepcave_run, history, incumbent

    def _load_metrics(self, fold_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        metrics_path = fold_dir / "validation" / VALIDATION_METRICS_FILENAME
        dataset_info_path = fold_dir / DATASET_JSON_FILENAME

        validation_metrics = load_json(metrics_path)
        dataset_info = load_json(dataset_info_path)
        labels = list(dataset_info["labels"].keys())

        foreground_mean = pd.DataFrame(validation_metrics["foreground_mean"], index=[0])

        mean = []
        for class_id, metrics in validation_metrics["mean"].items():
            row = {
                "class_id": class_id,
                "label": labels[int(class_id)],
                **metrics
            }
            mean.append(row)
        mean = pd.DataFrame(mean)

        metrics_per_case = []
        for case in validation_metrics["metric_per_case"]:
            prediction_file = Path(case["prediction_file"]).name
            reference_file = Path(case["reference_file"]).name

            for class_id, metrics in case["metrics"].items():
                row = {
                    "class_id": class_id,
                    "label": labels[int(class_id)],
                    **metrics,
                    "prediction_file": prediction_file,
                    "reference_file": reference_file
                }
                metrics_per_case.append(row)
        metrics_per_case = pd.DataFrame(metrics_per_case)

        return foreground_mean, mean, metrics_per_case

    def _load_progress(self, fold_dir: Path) -> pd.DataFrame:
        dataset_info = load_json(fold_dir / DATASET_JSON_FILENAME)

        labels = list(dataset_info["labels"].keys())
        if labels[0] == "background":
            labels = labels[1:]

        path = fold_dir / PROGRESS_FILENAME

        progress = pd.read_csv(path)
        progress["Epoch"] = np.arange(len(progress))

        progress["dice_per_class_or_region"] = progress[
            "dice_per_class_or_region"].apply(
                ast.literal_eval)
        labels_df = pd.DataFrame(
            progress["dice_per_class_or_region"].tolist(),
            columns=labels
            )

        # Use epoch_start_timestamps and epoch_end_timestamps
        progress["Runtime"] = progress["epoch_end_timestamps"] - progress["epoch_start_timestamps"]
        progress = progress.drop(columns=["epoch_start_timestamps", "epoch_end_timestamps"])

        return pd.concat(
            [progress, labels_df],
            axis=1
        ).drop(columns=["dice_per_class_or_region"])


    def _load_baseline_data(self, datasets: list[str]) -> BaselineResult:
        all_progress  = []
        all_emissions = []
        all_metrics_foreground_mean = []
        all_metrics_mean = []
        all_metrics_per_case = []

        for approach, baseline_dir in zip(
            list(APPROACH_REPLACE_MAP.values())[:3],
            [self.baseline_conv, self.baseline_resenc_m, self.baseline_resenc_l], strict=False
        ):
            for dataset in datasets:
                if dataset in self._baseline_datasets:
                    continue

                dataset_dir = baseline_dir / dataset
                if not dataset_dir.exists():
                    self.logger.info(f"{approach}: Skipping {dataset}.")
                    continue

                for fold in range(self.n_folds):
                    fold_dir = dataset_dir \
                        / self.configuration / f"fold_{fold}"
                    if not (fold_dir / "validation" / "summary.json").exists():
                        self.logger.info(f"{approach}: Skipping {fold} of dataset {dataset}.")
                        continue

                    progress = self._load_progress(fold_dir=fold_dir)

                    if (fold_dir / EMISSIONS_FILENAME).is_file():
                        emissions = pd.read_csv(fold_dir / EMISSIONS_FILENAME)
                    else:
                        emissions = pd.DataFrame()
                    (
                        metrics_foreground_mean,
                        metrics_mean,
                        metrics_per_case
                    ) = self._load_metrics(fold_dir)

                    for df in [progress, emissions, metrics_foreground_mean, metrics_mean, metrics_per_case]:
                        df["Approach"] = approach
                        df["Fold"] = fold
                        df["Dataset"] = dataset

                    all_progress.append(progress)
                    all_emissions.append(emissions)
                    all_metrics_foreground_mean.append(metrics_foreground_mean)
                    all_metrics_mean.append(metrics_mean)
                    all_metrics_per_case.append(metrics_per_case)

        all_progress = pd.concat(all_progress)
        all_emissions = pd.concat(all_emissions)
        all_metrics_foreground_mean = pd.concat(all_metrics_foreground_mean)
        all_metrics_mean = pd.concat(all_metrics_mean)
        all_metrics_per_case = pd.concat(all_metrics_per_case)

        return BaselineResult(
            progress=all_progress,
            metrics_foreground_mean=all_metrics_foreground_mean,
            metrics_mean=all_metrics_mean,
            metrics_per_case=all_metrics_per_case,
            emissions=all_emissions
        )

    def _load_incumbent(
            self,
            run_path: Path,
            filename: str = INCUMBENT_FILENAME,
            objective: str = "o0_loss"
        ) -> pd.DataFrame:
        incumbent = pd.read_csv(run_path / filename)
        history = pd.read_csv(run_path / HISTORY_FILENAME)

        # Since we run succesive halving, we have to insert the real
        # budget of a run by subtracting the budget of the previous run
        # in the runhistory
        history.loc[:, "real_budget"] = 0.
        for run_id, real_budget in self.real_budgets_per_config.items():
            history.loc[history["run_id"] == run_id, "real_budget"] = real_budget

        # The real used budget is the sum of all additional budgets
        incumbent["real_budget_used"] = history["real_budget"].cumsum()

        assert len(incumbent) == len(history)

        # Now we exapand the incumbent to have one row per fold
        incumbent_expanded = []
        for _, row in incumbent.iterrows():
            for fold, performance_key in enumerate(
                [f"{objective}_fold_{i}" for i in range(5)]
            ):
                performance = history[
                    history["run_id"] == row["run_id"]
                ][performance_key].to_numpy()[0]

                row_data = {
                    "Run ID": row["run_id"],
                    "Configuration ID": row["config_id"],
                    "1 - Dice": performance,
                    "Budget": row["budget"],
                    "Fold": fold,
                    "Budget Used": row["budget_used"],
                    "Real Budget Used": row["real_budget_used"],
                }

                incumbent_expanded.append(row_data)

        incumbent = pd.DataFrame(incumbent_expanded)

        # In addition to the hypersweeper incubment, we also log the
        # config origins manually in NePS. Now we want to merge them
        if origins := self._load_sampling_policy(run_path / SAMPLING_POLICY_LOGS):
            incumbent.loc[:, "Config Origin"] = ""   
            for run_id, origin in origins.items():
                incumbent.loc[incumbent["Configuration ID"] == run_id, "Config Origin"] = origin    

        incumbent["Full Model Trainings"] = incumbent["Real Budget Used"] / self.max_budget

        return incumbent
    
    def _load_sampling_policy(self, log_path: Path) -> dict[int, str] | None:
        if not log_path.exists():
            return None
        
        with open(log_path, "r") as file:
            lines = file.readlines()

        lines = [line.strip() for line in lines]

        # In case the optimization was re-run due to some job crashes,
        # we only keep the origins from the last full run
        if len(lines) > 128:
            lines = lines[-128:]

        origins = {i + 1: ORIGIN_MAP[origin] for i, origin in enumerate(lines)}
        origins[0] = "Default"

        return origins

    def _load_history(self, run_path: Path) -> pd.DataFrame:
        history = pd.read_csv(run_path / HISTORY_FILENAME)
        history = history.rename(columns=HISTORY_REPLACEMENT_MAP)

        # In addition to the hypersweeper history, we also log the
        # config origins manually in NePS. Now we want to merge them
        if origins := self._load_sampling_policy(run_path / SAMPLING_POLICY_LOGS):
            history.loc[:, "Config Origin"] = ""   
            for run_id, origin in origins.items():
                history.loc[history["Configuration ID"] == run_id, "Config Origin"] = origin    

        return history

    def _load_hpo_data(self, datasets: list[str]) -> HPOResult:
        all_progress  = []
        all_emissions = []
        all_history = []
        all_incumbent = []
        deepcave_runs = {}

        approach_key = "hpo"
        approach = APPROACH_REPLACE_MAP[approach_key]

        for dataset in datasets:
            if dataset in self._hpo_datasets:
                continue
            
            dataset_dir = self.hpo_dir / dataset
            if not dataset_dir.exists():
                self.logger.info(f"{approach}: Skipping {dataset}.")
                continue

            hpo_run_dir = dataset_dir \
                / self.configuration / str(self.hpo_seed)
            if not (hpo_run_dir / HISTORY_FILENAME).exists():
                self.logger.info(f"{approach}: Skipping {dataset}.")
                continue

            history = self._load_history(hpo_run_dir)
            history["Dataset"] = dataset
            history["Approach"] = approach

            deepcave_runs[dataset] = runhistory_to_deepcave(
                dataset=dataset,
                history=history,
                approach_key=approach_key
            )

            incumbent = self._load_incumbent(
                run_path=hpo_run_dir
            )
            incumbent["Dataset"] = dataset
            incumbent["Approach"] = approach

            all_history.append(history)
            all_incumbent.append(incumbent)

            incumbent_run_id = incumbent["Run ID"].to_numpy()[-1]

            for run_id in history["Run ID"].unique():
                for fold in range(self.n_folds):
                    slurm_run_id = run_id * self.n_folds + fold
                    run_dir = hpo_run_dir / str(slurm_run_id)

                    if not (run_dir).exists():
                        print(f"Unable to read {run_dir}.")
                        continue

                    if (run_dir / EMISSIONS_FILENAME).is_file():
                        emissions = pd.read_csv(run_dir / EMISSIONS_FILENAME)
                        emissions["Dataset"] = dataset
                        emissions["Approach"] = approach
                        emissions["Fold"] = fold
                        emissions["Run ID"] = run_id
                        all_emissions.append(emissions)

                    if run_id == incumbent_run_id:
                        progress = self._load_progress(run_dir)
                        progress["Dataset"] = dataset
                        progress["Approach"] = approach
                        progress["Fold"] = fold
                        progress["Run ID"] = run_id

                        all_progress.append(progress)

        all_progress = pd.concat(all_progress)
        all_emissions = pd.concat(all_emissions)
        all_history = pd.concat(all_history)
        all_incumbent = pd.concat(all_incumbent)

        return HPOResult(
            history=all_history,
            incumbent=all_incumbent,
            incumbent_progress=all_progress,
            emissions=all_emissions,
            deepcave_runs=deepcave_runs
        )

    def _load_nas_data(self, datasets: list[str], approach_key: str) -> NASResult:
        all_emissions = []
        all_history = []
        all_incumbent = defaultdict(list)
        deepcave_runs = {}
        
        if approach_key == "hpo_nas":
            base_dir = self.nas_dir 
            dataset_list = self._nas_datasets
        elif approach_key == "hpo_hnas":
            base_dir = self.hnas_dir
            dataset_list = self._hnas_datasets
        else:
            raise ValueError(f"Unknown approach key {approach_key}.")

        approach = APPROACH_REPLACE_MAP[approach_key]

        for dataset in datasets:
            if dataset in dataset_list:
                continue
    
            dataset_dir = base_dir / dataset
            if not dataset_dir.exists():
                self.logger.info(f"{approach}: Skipping {dataset}.")
                continue

            nas_run_dir = dataset_dir \
                / self.configuration / str(self.hpo_seed)
            if not (nas_run_dir / HISTORY_FILENAME).exists():
                self.logger.info(f"{approach}: Skipping {dataset}.")
                continue

            history = self._load_history(nas_run_dir)
            history["Dataset"] = dataset
            history["Approach"] = approach

            for i, objective in enumerate(self.objectives):
                incumbent_filename = f"incumbent_{OBJECTIVES_MAPPING[objective]}.csv"
                obj_name = f"o{i}_{OBJECTIVES_MAPPING[objective]}"

                incumbent = self._load_incumbent(
                    run_path=nas_run_dir,
                    filename=incumbent_filename,
                    objective=obj_name
                )
                incumbent["Dataset"] = dataset
                incumbent["Approach"] = approach

                all_incumbent[objective].append(incumbent)

            all_history.append(history)

            deepcave_runs[dataset] = runhistory_to_deepcave(
                dataset=dataset,
                history=history,
                approach_key=approach_key
            )

            for run_id in history["Run ID"].unique():
                for fold in range(self.n_folds):
                    slurm_run_id = run_id * self.n_folds + fold
                    run_dir = nas_run_dir / str(slurm_run_id)

                    if not (run_dir).exists():
                        print(f"Unable to read {run_dir}.")
                        continue

                    if (run_dir / EMISSIONS_FILENAME).is_file():
                        emissions = pd.read_csv(run_dir / EMISSIONS_FILENAME)
                        emissions["Dataset"] = dataset
                        emissions["Approach"] = approach
                        emissions["Fold"] = fold
                        emissions["Run ID"] = run_id
                        all_emissions.append(emissions)

        all_emissions = pd.concat(all_emissions)
        all_history = pd.concat(all_history)

        all_incumbent_df = {}
        for objective in self.objectives:
            all_incumbent_df[objective] = pd.concat(all_incumbent[objective])

        return NASResult(
            history=all_history,
            incumbents=all_incumbent_df,
            emissions=all_emissions,
            deepcave_runs=deepcave_runs
        )

    def _enable_grid(self, ax: Any) -> None:
        ax.minorticks_on()
        ax.grid(which="major", linestyle="-", linewidth=0.5, alpha=0.7)
        ax.grid(which="minor", linestyle="--", linewidth=0.5, alpha=0.7)

    def _plot_baseline(self, dataset: str, log_x: bool = True) -> None:
        baseline_progress = self.get_baseline_data(dataset).progress

        # smoothing the curves for each approach
        baseline_progress.loc[:, "mean_fg_dice"] = baseline_progress.groupby(
            ["Approach", "Fold"])["mean_fg_dice"].transform(
                lambda x: x.ewm(span=10).mean()
            )
        fig, ax = plt.subplots(1, 1, figsize=(self.figwidth / 2, self.figwidth / 2))

        baseline_progress.loc[:, "Approach"] = baseline_progress["Approach"].replace({
            "nnU-Net (Conv)": "Conv",
            "nnU-Net (ResM)": "ResM",
            "nnU-Net (ResL)": "ResL"
        })

        g = sns.lineplot(
            x="Epoch",
            y="mean_fg_dice",
            data=baseline_progress,
            hue="Approach",
            errorbar=("sd")
        )

        g.set_title(f"nnU-Net Training Progress on\n{format_dataset_name(dataset)}")
        g.set_xlabel("Epoch")
        g.set_ylabel("Mean Foreground Dice (Proxy)")

        if log_x:
            g.set_xscale("log")
            g.set_xlim(1, 1000)

        g.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.17),
            ncol=3,
            fancybox=False,
            shadow=False,
            frameon=False
        )
        self._enable_grid(g)

        fig.subplots_adjust(
            top=0.88,
            bottom=0.2,
            left=0.18,
            right=0.96,
        )
        plt.savefig(
            self.baseline_plots / f"{dataset}.{self.format}",
            format=self.format,
            dpi=self.dpi
        )
        plt.clf()

    def plot_baselines(
            self,
        ):
        for dataset in self.datasets:
            self._plot_baseline(dataset)

    def _plot_optimization(
            self,
            dataset: str,
            x_log_scale: bool = False,
            y_log_scale: bool = False,
            include_nas: bool = True,
            include_hnas: bool = True,
            show_error: bool = False
        ) -> None:
        color_palette = self.color_palette[:5] + [self.color_palette[9]]

        fig, ax = plt.subplots(1, 1, figsize=(self.figwidth / 2, self.figwidth / 2))

        hpo_data = self.get_hpo_data(dataset)
        incumbent = hpo_data.incumbent

        if include_nas:
            nas_data = self.get_nas_data(dataset)
            incumbent = pd.concat([incumbent, nas_data.incumbents["1 - Dice"]])
        if include_hnas:
            hnas_data = self.get_hnas_data(dataset)
            incumbent = pd.concat([incumbent, hnas_data.incumbents["1 - Dice"]])

        baseline_data = self.get_baseline_data(dataset)

        metrics = baseline_data.metrics_foreground_mean

        metrics_expanded = pd.DataFrame(
            np.repeat(metrics.values, 2, axis=0),
            columns=metrics.columns
        )
        metrics_expanded["Full Model Trainings"] = np.tile(
            [0, self.show_n_full_trainings],
            len(metrics)
        )
        metrics_expanded.loc[:, "1 - Dice"] = (1 - metrics_expanded["Dice"])

        n_hpo_approaches = len(incumbent["Approach"].unique())
        n_baseline_approaches = len(metrics_expanded["Approach"].unique())

        g = sns.lineplot(
            data=metrics_expanded,
            x="Full Model Trainings",
            y="1 - Dice",
            hue="Approach",
            palette=self.color_palette[:min(3, n_baseline_approaches)],
            linestyle="--",
            errorbar=("sd") if show_error else None,
        )

        sns.lineplot(
            x="Full Model Trainings",
            y="1 - Dice",
            data=incumbent,
            drawstyle="steps-post",
            hue="Approach",
            errorbar=("sd") if show_error else None,
            palette=self.color_palette[3: n_hpo_approaches + 3],
            ax=g
        )

        # We add markers to highlight the final values
        hpo_approaches = incumbent["Approach"].unique()
        for i in range(n_hpo_approaches):
            approach = hpo_approaches[i]
            grouped_approach = incumbent[incumbent["Approach"] == approach].groupby("Full Model Trainings")
            last_value = grouped_approach["1 - Dice"].mean().iloc[-1]
            last_x = grouped_approach["Full Model Trainings"].mean().iloc[-1]

            color = color_palette[3 + i]
            ax.scatter(
                last_x,
                last_value,
                color=color,
                zorder=5,
                linewidth=1,
                s=100,
                marker="*",
                edgecolor="black"
            )

        g.set_title(format_dataset_name(dataset))
        g.set_xlabel("Full Model Trainings")
        g.set_ylabel("1 - Dice")

        if x_log_scale:
            g.set_xscale("log")
            g.set_xlim(0.1, self.show_n_full_trainings)
        else:
            g.set_xlim(0, self.show_n_full_trainings)

        if y_log_scale:
            g.set_ylim(1e-2, 1)
            g.set_yscale("log")
        else:
            pass
        
        self._enable_grid(g)

        g.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=2,
            fancybox=False,
            shadow=False,
            frameon=False
        )

        fig.subplots_adjust(
            top=0.92,
            bottom=0.3,
            left=0.18,
            right=0.94,
        )

        plt.savefig(
            self.hpo_plots / f"{dataset}_{'hpo_nas' if include_nas or include_hnas else 'hpo'}.{self.format}",
            format=self.format,
            dpi=self.dpi
        )
        plt.clf()


    def plot_hpo(self, **kwargs) -> None:
        for dataset in self._hpo_datasets:
            try:
                self._plot_optimization(dataset, **kwargs)
            except ValueError:
                self.logger.info(f"Unable to plot HPO for {dataset}.")
                continue

    def plot_optimization_combined(
            self,
            x_log_scale: bool = False,
            y_log_scale: bool = False,
            include_nas: bool = True,
            include_hnas: bool = True,
            show_error: bool = False
        ) -> None:
        color_palette = self.color_palette[:5] + [self.color_palette[9]]

        fig, axes = plt.subplots(
            nrows=2,
            ncols=5,
            sharex=True,
            figsize=(self.figwidth, 4)
        )
        axes = axes.flatten()

        max_baseline_approaches = 0
        max_hpo_approaches = 0
        max_baseline_approaches_ax = axes[0]
        max_hpo_approaches_ax = axes[0]

        for ax, dataset in zip(axes, self._hpo_datasets, strict=False):
            hpo_data = self.get_hpo_data(dataset)
            incumbent = hpo_data.incumbent

            if include_nas and dataset in self._nas_datasets:
                nas_data = self.get_nas_data(dataset)
                incumbent = pd.concat([incumbent, nas_data.incumbents["1 - Dice"]])
            if include_hnas and dataset in self._hnas_datasets:
                hnas_data = self.get_hnas_data(dataset)
                incumbent = pd.concat([incumbent, hnas_data.incumbents["1 - Dice"]])

            incumbent.loc[:, "1 - Dice"] *= 100

            baseline_data = self.get_baseline_data(dataset)

            metrics = baseline_data.metrics_foreground_mean

            metrics_expanded = pd.DataFrame(
                np.repeat(metrics.values, 2, axis=0),
                columns=metrics.columns
            )
            metrics_expanded["Full Model Trainings"] = np.tile(
                [0, self.show_n_full_trainings],
                len(metrics)
            )
            metrics_expanded.loc[:, "1 - Dice"] = (1 - metrics_expanded["Dice"]) * 100

            n_hpo_approaches = len(incumbent["Approach"].unique())
            n_baseline_approaches = len(metrics_expanded["Approach"].unique())

            if n_baseline_approaches > max_baseline_approaches:
                max_baseline_approaches = n_baseline_approaches
                max_baseline_approaches_ax = ax

            if n_hpo_approaches > max_hpo_approaches:
                max_hpo_approaches = n_hpo_approaches
                max_hpo_approaches_ax = ax

            # Plot baselines
            g = sns.lineplot(
                data=metrics_expanded,
                x="Full Model Trainings",
                y="1 - Dice",
                hue="Approach",
                palette=color_palette[:min(3, n_baseline_approaches)],
                linestyle="--",
                errorbar=("sd") if show_error else None,
                ax=ax
            )

            # Plot our approaches
            sns.lineplot(
                x="Full Model Trainings",
                y="1 - Dice",
                data=incumbent,
                drawstyle="steps-post",
                hue="Approach",
                errorbar=("sd") if show_error else None,
                palette=color_palette[3: n_hpo_approaches + 3],
                ax=ax
            )

            # We add markers to highlight the final values
            hpo_approaches = incumbent["Approach"].unique()
            for i in range(n_hpo_approaches):
                approach = hpo_approaches[i]
                grouped_approach = incumbent[incumbent["Approach"] == approach].groupby("Full Model Trainings")
                last_value = grouped_approach["1 - Dice"].mean().iloc[-1]
                last_x = grouped_approach["Full Model Trainings"].mean().iloc[-1]

                color = color_palette[3 + i]
                ax.scatter(
                    last_x,
                    last_value,
                    color=color,
                    zorder=5,
                    linewidth=1,
                    s=100,
                    marker="*",
                    edgecolor="black"
                )

            g.set_title(format_dataset_name(dataset).replace(" ", "\n"))
            g.set_xlabel("Full Model Trainings")
            g.set_ylabel("1 - Dice [%]")

            if x_log_scale:
                g.set_xscale("log")
                g.set_xlim(0.1, self.show_n_full_trainings)
            else:
                g.set_xlim(0, self.show_n_full_trainings)
                g.set_xticks(range(0, self.show_n_full_trainings + 1, 5))

            if y_log_scale:
                g.set_ylim(1e-2, 1)
                g.set_yscale("log")

            if ax != axes[0] and ax != axes[5]:
                ax.set_ylabel("")

            ax.get_legend().remove()

        # We use the axes with most approaches to get the legend
        baseline_handles, baseline_labels = max_baseline_approaches_ax.get_legend_handles_labels()
        hpo_handles, hpo_labels = max_hpo_approaches_ax.get_legend_handles_labels()
        
        zipped_handles = [val for pair in zip(baseline_handles[:max_baseline_approaches], hpo_handles[-max_hpo_approaches:]) for val in pair]
        zipped_labels = [val for pair in zip(baseline_labels[:max_baseline_approaches], hpo_labels[-max_hpo_approaches:]) for val in pair]

        fig.subplots_adjust(
            top=0.88,
            bottom=0.24,
            left=0.09,
            right=0.975,
            hspace=0.42,
            wspace=0.49
        )

        axes[-3].legend(
            zipped_handles,
            zipped_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=3,
            fancybox=False,
            shadow=False,
            frameon=False
        )

        plt.savefig(
            self.hpo_plots / "hpo_combined.png" if not include_nas else self.hpo_plots / f"hpo_nas_combined.{self.format}",
            format=self.format,
            dpi=self.dpi
        )
        plt.clf()

    def _plot_nas_budgets(
            self,
            dataset: str,
            approach_key: str,
            show_configs: bool = True,
        ) -> None:
        if approach_key == "hpo_nas":
            nas_data = self.get_nas_data(dataset)
        elif approach_key == "hpo_hnas":
            nas_data = self.get_hnas_data(dataset)
        else:
            raise ValueError(f"Unknown approach key {approach_key}.")
        
        baseline_data = self.get_baseline_data(dataset)

        fig, ax = plt.subplots(1, 1, figsize=(self.figwidth / 2, self.figwidth / 2))

        n_baseline_approaches = 0

        for baseline_approach in baseline_data.metrics_foreground_mean["Approach"].unique():
            metrics = baseline_data.metrics_foreground_mean
            metrics = metrics[metrics["Approach"] == baseline_approach]
            baseline_dice = metrics["Dice"].mean() * 100

            baseline_progress = baseline_data.progress
            baseline_progress = baseline_progress[baseline_progress["Approach"] == baseline_approach]
            baseline_time = baseline_progress.groupby("Fold")["Runtime"].sum().mean() / 3600

            color = self.color_palette[n_baseline_approaches]

            sns.scatterplot(
                x=[100 - baseline_dice],
                y=[baseline_time],
                color=color,
                label=baseline_approach,
                marker="x",
                ax=ax,
                linewidth=2,
                zorder=4
            )

            n_baseline_approaches += 1

        hpo_data = self.get_hpo_data(dataset)

        # get mean of last 5 entries in incumbent
        hpo_dice = hpo_data.incumbent["1 - Dice"].iloc[-5:].mean() * 100
        hpo_time = hpo_data.incumbent_progress.groupby("Fold")["Runtime"].sum().mean() / 3600

        color = self.color_palette[n_baseline_approaches]

        sns.scatterplot(
            x=[hpo_dice],
            y=[hpo_time],
            color=color,
            label="HPO (ours)",
            marker="x",
            ax=ax,
            linewidth=2,
            zorder=4
        )

        history = nas_data.history
        history.loc[:, "1 - Dice"] *= 100

        pareto_front = history.sort_values(by="1 - Dice")
        pareto_front = pareto_front[pareto_front["Runtime"] == pareto_front["Runtime"].cummin()]

        g = sns.lineplot(
            data=pareto_front,
            x="1 - Dice",
            y="Runtime",
            color=self.color_palette[9],
            label="HPO + NAS (ours)",
            ax=ax,
            drawstyle="steps-post",
            zorder=3
        )

        if show_configs:
            # Round budget column
            history.loc[:, "Budget"] = history["Budget"].round()
            g = sns.scatterplot(
                data=history,
                x="1 - Dice",
                y="Runtime",
                # label="Configurations",
                color=self.color_palette[n_baseline_approaches + 1],
                size="Budget",
                s=10,
                alpha=0.5,
                ax=ax,
            )

        g.set_title(format_dataset_name(dataset))
        g.set_xlabel("1 - Dice")
        g.set_ylabel("Training Runtime [h]")

        self._enable_grid(g)

        g.set_xscale("log")
        g.set_yscale("log")

        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=2,
            fancybox=False,
            shadow=False,
            frameon=False
        )

        fig.subplots_adjust(
            top=0.94,
            bottom=0.39,
            left=0.18,
            right=0.96,
        )

        if approach_key == "hpo_nas":
            dir = self.nas_plots
        else:
            dir = self.hnas_plots
        plt.savefig(
            dir / f"{dataset}_{f'nas' if show_configs else f'nas_no_configs'}.{self.format}",
            format=self.format,
            dpi=self.dpi
        )
        plt.clf()

    def _plot_nas_origins(
            self,
            dataset: str,
            approach_key: str,
        ) -> None:
        if approach_key == "hpo_nas":
            nas_data = self.get_nas_data(dataset)
        elif approach_key == "hpo_hnas":
            nas_data = self.get_hnas_data(dataset)
        else:
            raise ValueError(f"Unknown approach key {approach_key}.")
        
        fig, ax = plt.subplots(1, 1, figsize=(self.figwidth / 2, 3))

        history = nas_data.history
        history.loc[:, "1 - Dice"] *= 100

        pareto_front = history.sort_values(by="1 - Dice")
        pareto_front = pareto_front[pareto_front["Runtime"] == pareto_front["Runtime"].cummin()]

        g = sns.lineplot(
            data=pareto_front,
            x="1 - Dice",
            y="Runtime",
            color=self.color_palette[0],
            label="HPO + NAS (ours)",
            ax=ax,
            drawstyle="steps-post",
            zorder=3
        )

        origins = [origin for origin in history["Config Origin"].unique() if origin != "Default"]
        markers = ["x", "x", "x"]
        origin_markers = {origin: marker for origin, marker in zip(origins, markers)}
        colors = self.color_palette[1:4]
        origin_colors = {origin: color for origin, color in zip(origins, colors)}

        for origin in origins:
            subset = history[history["Config Origin"] == origin]
            marker = origin_markers[origin]
            color = origin_colors[origin]
            g = sns.scatterplot(
                data=subset,
                x="1 - Dice",
                y="Runtime",
                label=origin,
                color=color,
                marker=marker,
                s=15,
                ax=ax,
                linewidth=1,
            )

        g.set_title(format_dataset_name(dataset))
        g.set_xlabel("1 - Dice")
        g.set_ylabel("Training Runtime [h]")

        self._enable_grid(g)

        g.set_xscale("log")
        g.set_yscale("log")

        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=2,
            fancybox=False,
            shadow=False,
            frameon=False
        )

        fig.subplots_adjust(
            top=0.92,
            bottom=0.3,
            left=0.18,
            right=0.96,
        )

        if approach_key == "hpo_nas":
            dir = self.nas_plots
        else:
            dir = self.hnas_plots
        plt.savefig(
            dir / f"{dataset}_nas_origins.{self.format}",
            format=self.format,
            dpi=self.dpi
        )
        plt.clf()

    def plot_nas_budgets(self, approach_key: str, **kwargs) -> None:
        for dataset in self.datasets:
            self._plot_nas_budgets(dataset, approach_key=approach_key, **kwargs)

    def plot_nas_origins(self, approach_key: str, **kwargs) -> None:
        for dataset in self.datasets:
            self._plot_nas_origins(dataset, approach_key=approach_key, **kwargs)

    def plot_nas_combined(
            self,
            x_log_scale: bool = True,
            y_log_scale: bool = True
        ) -> None:

        fig, axes = plt.subplots(
            nrows=2,
            ncols=5,
            sharex=False,
            sharey=True,
            figsize=(12, 6)
        )
        axes = axes.flatten()

        max_approaches_ax = axes[0]

        for ax, dataset in zip(axes, self._hpo_datasets, strict=False):
            nas_data = self.get_nas_data(dataset)
            hnas_data = self.get_hnas_data(dataset)
            baseline_data = self.get_baseline_data(dataset)

            n_baseline_approaches = 0

            for baseline_approach in baseline_data.metrics_foreground_mean["Approach"].unique():
                metrics = baseline_data.metrics_foreground_mean
                metrics = metrics[metrics["Approach"] == baseline_approach]
                baseline_dice = metrics["Dice"].mean()

                baseline_progress = baseline_data.progress
                baseline_progress = baseline_progress[baseline_progress["Approach"] == baseline_approach]
                baseline_time = baseline_progress.groupby("Fold")["Runtime"].sum().mean() / 3600

                color = self.color_palette[n_baseline_approaches]

                sns.scatterplot(
                    x=[(1 - baseline_dice) * 100],
                    y=[baseline_time],
                    color=color,
                    label=baseline_approach,
                    ax=ax
                )

                n_baseline_approaches += 1

            hpo_data = self.get_hpo_data(dataset)

            # get mean of last 5 entries in incumbent
            hpo_dice = hpo_data.incumbent["1 - Dice"].iloc[-5:].mean() * 100
            hpo_time = hpo_data.incumbent_progress.groupby("Fold")["Runtime"].sum().mean() / 3600

            color = self.color_palette[n_baseline_approaches]

            sns.scatterplot(
                x=[hpo_dice],
                y=[hpo_time],
                color=color,
                label="HPO (ours)",
                marker="x",
                ax=ax
            )

            nas_history = nas_data.history
            hnas_history = hnas_data.history

            nas_pareto_front = nas_history.sort_values(by="1 - Dice")
            nas_pareto_front = nas_pareto_front[nas_pareto_front["Runtime"] == nas_pareto_front["Runtime"].cummin()]

            hnas_pareto_front = hnas_history.sort_values(by="1 - Dice")
            hnas_pareto_front = hnas_pareto_front[hnas_pareto_front["Runtime"] == hnas_pareto_front["Runtime"].cummin()]

            nas_pareto_front.loc[:, "1 - Dice"] *= 100
            hnas_pareto_front.loc[:, "1 - Dice"] *= 100

            sns.lineplot(
                data=nas_pareto_front,
                x="1 - Dice",
                y="Runtime",
                color=self.color_palette[n_baseline_approaches + 1],
                label="HPO + NAS (ours)",
                ax=ax,
                drawstyle="steps-post"
            )

            sns.lineplot(
                data=hnas_pareto_front,
                x="1 - Dice",
                y="Runtime",
                color=self.color_palette[n_baseline_approaches + 2],
                label="HPO + HNAS (ours)",
                ax=ax,
                drawstyle="steps-post"
            )

            ax.set_title(format_dataset_name(dataset).replace(" ", "\n"))
            ax.set_xlabel("1 - Dice [%]")
            ax.set_ylabel("Training Runtime [h]")

            ax.set_xscale("log")
            ax.set_yscale("log")

            self._enable_grid(ax)

            # xticks = ax.get_xticks()
            # xticklabels = [f"{int(x)}" for x in xticks]
            # if len(xticklabels) > 2:
            #     xticklabels = [xticklabels[0]] + ["" for _ in range(len(xticklabels) - 2)]+ [xticklabels[-1]]

            # ax.set_xticks(xticks)
            # ax.set_xticklabels(xticklabels)

            if ax != axes[0] and ax != axes[5]:
                ax.set_ylabel("")

            ax.get_legend().remove()

        # We use the axis with most approaches to get the legend
        handles, labels = max_approaches_ax.get_legend_handles_labels()

        zipped_handles = [val for pair in zip(handles[:n_baseline_approaches], handles[-3:]) for val in pair]
        zipped_labels = [val for pair in zip(labels[:n_baseline_approaches], labels[-3:]) for val in pair]

        fig.subplots_adjust(
            top=0.91,
            bottom=0.15,
            left=0.07,
            right=0.98,
            hspace=0.54,
            wspace=0.15
        )

        axes[-3].legend(
            zipped_handles,
            zipped_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.23),
            ncol=3,
            fancybox=False,
            shadow=False,
            frameon=False
        )

        plt.savefig(
            self.nas_plots / f"nas_combined.{self.format}",
            format=self.format,
            dpi=self.dpi
        )
        plt.clf()

    @staticmethod
    def _get_budget(budget: int, deepcave_run: DeepCAVERun) -> float | int:
        # DeepCAVE supports the use of COMBINED_BUDGET
        if budget == COMBINED_BUDGET:
            return budget

        # Now, an index is passed, so we need to get the actual
        # budget for that index
        budgets = deepcave_run.get_budgets()
        if budget not in range(len(budgets) - 1):
            raise ValueError(f"Budget index {budget} not found in {budgets}.")
        return float(budgets[budget])
        
    def _plot_hp_importances(
            self,
            budget: int = COMBINED_BUDGET,
            method: Literal["global", "local"] = "global"
        ):
        fig, axes = plt.subplots(
            nrows=2,
            ncols=5,
            sharex=True,
            sharey=True,
            figsize=(self.figwidth, 5)
        )
        axes = axes.flatten()

        hyperparameters = {k: v for k, v in HYPERPARAMETER_REPLACEMENT_MAP.items() if k in HPO_HYPERPARAMETERS}

        for ax, dataset in zip(axes, self.datasets, strict=False):
            deepcave_run = self.get_hpo_data(dataset).deepcave_runs[dataset]

            selected_budget = self._get_budget(budget, deepcave_run)

            if method == "global":
                evaluator = fANOVA(run=deepcave_run)
                evaluator.calculate(budget=selected_budget, seed=42)
            else:
                evaluator = LPI(run=deepcave_run)
                evaluator.calculate(budget=selected_budget, seed=42)
            
            importances = evaluator.get_importances(hp_names=list(deepcave_run.configspace.keys()))

            importances_data = []
            for hp_key, hp_name in hyperparameters.items():
                importance = importances[hp_key]
                importances_data += [{
                    "Hyperparameter": hp_name,
                    "Importance": importance[0],
                    "Error": importance[1],
                }]
            importances_df = pd.DataFrame(importances_data)

            g = sns.barplot(
                data=importances_df,
                x="Importance",
                y="Hyperparameter",
                hue="Hyperparameter",
                hue_order=hyperparameters.values(),
                ax=ax,
                errorbar=None,
                dodge=False,
                legend="full"
            )

            g.errorbar(
                y=importances_df["Hyperparameter"],
                x=importances_df["Importance"],
                xerr=importances_df["Error"],
                fmt="none",
                c="black",
                capsize=3,
            )

            g.set_title(format_dataset_name(dataset).replace(" ", "\n"))
            g.set_yticklabels([])

            ax.legend().set_visible(False)

        axes[-1].legend().set_visible(False)

        fig.subplots_adjust(
            top=0.85,
            bottom=0.17,
            left=0.04,
            right=0.98,
            hspace=0.3,
            wspace=0.2
        )

        # Create a single legend
        axes[-3].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=4,
            fancybox=False,
            shadow=False,
            frameon=False
        )

        _method = "Global" if method == "global" else "Local"
        fig.suptitle(f"{_method} HPIs for HPO")

        plt.savefig(
            self.hpo_analysis_plots / f"{method}_hpis.{self.format}",
            format=self.format,
            dpi=self.dpi
        )
        plt.clf()

    def _plot_mo_hp_importances(
            self,
            dataset: str,
            approach_key: str,
            budget: int = COMBINED_BUDGET,
            method: Literal["global", "local"] = "global"
        ):
        try:
            if approach_key == "hpo_nas":
                deepcave_run = self.get_nas_data(dataset).deepcave_runs[dataset]
            elif approach_key == "hpo_hnas":
                deepcave_run = self.get_hnas_data(dataset).deepcave_runs[dataset]
        except KeyError:
            return

        selected_budget = self._get_budget(budget, deepcave_run)

        if method == "global":
            evaluator = MOfANOVA(run=deepcave_run)
            evaluator.calculate(budget=selected_budget, seed=42)
            all_importances = evaluator.importances_
        else:
            evaluator = MOLPI(run=deepcave_run)
            evaluator.calculate(budget=selected_budget, seed=42)
            all_importances = evaluator.importances

        nas_approach = "HNAS" if approach_key == "hpo_hnas" else "NAS"
        nas_hp_names = NAS_HYPERPARAMETERS if approach_key == "hpo_nas" else HNAS_HYPERPARAMETERS
        
        if approach_key == "hpo_hnas":
            fig_height = 4.5
        else:
            fig_height = 4

        fig, axs = plt.subplots(
            nrows=1,
            ncols=2,
            sharex=True,
            figsize=(self.figwidth, fig_height)
        )

        assert all_importances is not None
        if len(all_importances) == 0:
            return

        for i, ax in enumerate(axs):
            if i == 0:
                # HPO
                importances = all_importances[all_importances["hp_name"].isin(HPO_HYPERPARAMETERS)]
                importances.loc[:, "hp_name"] = importances.loc[:, "hp_name"].replace(HYPERPARAMETER_REPLACEMENT_MAP)
                hp_names = [v for k, v in HYPERPARAMETER_REPLACEMENT_MAP.items() if k in HPO_HYPERPARAMETERS]
            else:
                # NAS or HNAS
                importances = all_importances[all_importances["hp_name"].isin(nas_hp_names)]
                importances.loc[:, "hp_name"] = importances.loc[:, "hp_name"].replace(HYPERPARAMETER_REPLACEMENT_MAP)
                hp_names = [v for k, v in HYPERPARAMETER_REPLACEMENT_MAP.items() if k in nas_hp_names]

            for j, hp in enumerate(hp_names):
                hp_data = importances[importances["hp_name"] == hp]
                hp_data = hp_data.sort_values(by="weight")
                x = hp_data["weight"]
                y = hp_data["importance"]
                variance = hp_data["variance"]

                color = self.color_palette[j % len(self.color_palette)]
                if j >= len(self.color_palette):
                    linestyle = "--"
                else:
                    linestyle = "-"

                lower = y - variance
                upper = y + variance

                ax.fill_between(
                    x,
                    lower,
                    upper,
                    color=color,
                    alpha=0.2
                )

                sns.lineplot(
                    x=x,
                    y=y,
                    ax=ax,
                    label=hp,
                    color=color,
                    linestyle=linestyle
                )

            ax.legend(
                title=f"{'HPO' if i == 0 else nas_approach} Hyperparameters", 
                bbox_to_anchor=(0.5, -0.35),
                loc="upper center",
                ncol=2,
                fancybox=False,
                shadow=False,
                frameon=False
            )
            ax.set_xlabel("Weight of 1 - Dice")
            ax.set_ylabel("Importance")

        axs[0].set_title("HPO Hyperparameters")
        axs[1].set_title(f"{nas_approach} Hyperparameters")

        # To imitate DeepCAVE, we add the objectives to the xticks
        xticks = ax.get_xticks() 
        xticklabels = ax.get_xticklabels()  
        xticklabels[1] = f"0.0\nRuntime"
        xticklabels[-2] = f"1.0\n1 - Dice"
        ax.set_xticks(xticks[1:-1])
        ax.set_xticklabels(xticklabels[1:-1])

        # As NAS and HNAS have different numbers of hyperparameters, we need to adjust the layout
        if approach_key == "hpo_hnas":
            fig.subplots_adjust(
                top=0.87,
                bottom=0.51,
                left=0.07,
                right=0.97,
                wspace=0.2,
                hspace=0.5,
            )
        else:
            fig.subplots_adjust(
                top=0.86,
                bottom=0.43,
                left=0.08,
                right=0.98,
                wspace=0.2,
                hspace=0.5,
            )

        _method = "Global" if method == "global" else "Local"
        fig.suptitle(f"{_method} MO-HPIs for HPO + {nas_approach} on {format_dataset_name(dataset)}")
        plt.grid(True)
        output_dir = self.analysis_plots[approach_key]
        plt.savefig(output_dir / f"{method}_mo_hpi_{dataset}.{self.format}",
            format=self.format,
            dpi=self.dpi
        )
        plt.clf()

    def plot_hpis(
            self,
            approach_keys: list[str] | None = None,
            budget: int = COMBINED_BUDGET,
        ):
        if approach_keys is None:
            approach_keys = APPROACHES[1:]
        for approach_key in approach_keys:
            if approach_key == "hpo":
                self._plot_hp_importances(budget=budget, method="global")
                self._plot_hp_importances(budget=budget, method="local")
            else:
                for dataset in self.datasets:
                    self._plot_mo_hp_importances(dataset=dataset, approach_key=approach_key, budget=budget, method="global")
                    self._plot_mo_hp_importances(dataset=dataset, approach_key=approach_key, budget=budget, method="local")

    def _plot_single_objective_ablation(
            self,
            performances: dict,
            improvements: dict
        ) -> matplotlib.figure.Figure:
        fig, axs = plt.subplots(1, 2, figsize=(self.figwidth, self.figwidth / 2))

        # 1) Performances
        hps = [HYPERPARAMETER_REPLACEMENT_MAP[hp] for hp in performances.keys()]

        perf_values = [(mean, var) for mean, var in performances.values()]
        impr_values = [(mean, var) for mean, var in improvements.values()]
        sns.lineplot(
            x=hps,
            y=[mean for mean, _ in perf_values],
            ax=axs[0],
            color=self.color_palette[0]
        )

        for hp, (mean, var) in zip(hps, perf_values):
            axs[0].errorbar(
                x=[hp],
                y=[mean],
                yerr=[var],
                fmt="o",
                capsize=5,
                color=self.color_palette[0]
            )

        # 2) Improvements with errorbar
        # Here, we skip the default configuration as it does
        # not have an improvement
        sns.barplot(
            x=hps[1:],
            y=[mean for mean, _ in impr_values[1:]],
            ax=axs[1],
            color=self.color_palette[0],
        )
        for hp, (mean, var) in zip(hps[1:], impr_values[1:]):
            axs[1].errorbar(
                x=[hp],
                y=[mean],
                yerr=[var],
                capsize=5,
                color="black"
            )

        # rotate x-axis labels
        for ax in axs:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")

        axs[0].set_title("Performance")
        axs[1].set_title("Improvement")

        axs[0].set_ylabel("1 - Dice")
        axs[1].set_ylabel("Improvement")

        axs[0].set_xlabel("Hyperparameter")
        axs[1].set_xlabel("Hyperparameter")

        fig.subplots_adjust(
            top=0.87,
            bottom=0.39,
            left=0.1,
            right=0.98,
            wspace=0.33,
            hspace=0.33
        )

        return fig


    def _plot_multi_objective_ablation(
            self,
            approach_key: str,
            importances: pd.DataFrame,
        ) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(self.figwidth, self.figwidth / 2))

        if approach_key == "hpo_hnas":
            fig_height = 4.5
        else:
            fig_height = 4

        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            sharex=True,
            figsize=(self.figwidth / 2, self.figwidth / 2)
        )

        importances.loc[:, "hp_name"] = importances.loc[:, "hp_name"].replace(HYPERPARAMETER_REPLACEMENT_MAP)
        importances.loc[importances["hp_name"] == "Default", "accuracy"] = 1 - importances["new_performance"]
        importances.loc[importances["hp_name"] != "Default", "accuracy"] = importances["importance"]

        grouped_importances = importances.groupby(["weight", "hp_name"])["accuracy"].sum().unstack(fill_value=0)

        colors = {column: self.color_palette[i] for i, column in enumerate(grouped_importances.columns)}

        cumulative = np.zeros(len(grouped_importances))
        for column in grouped_importances.columns:
            plt.fill_between(
                grouped_importances.index,
                cumulative,
                cumulative + grouped_importances[column],
                label=column,
                color=colors[column],
            )
            cumulative += grouped_importances[column]

        ax.legend(
            title="Hyperparameters", 
            bbox_to_anchor=(0.5, -0.28),
            loc="upper center",
            ncol=2,
            fancybox=False,
            shadow=False,
            frameon=False
        )

        ax.set_xlabel("Weight of 1 - Dice")
        ax.set_ylabel("Sum of weighted\nnormalized performance")

        # To imitate DeepCAVE, we add the objectives to the xticks
        xticks = ax.get_xticks() 
        xticklabels = ax.get_xticklabels()  
        xticklabels[1] = f"0.0\nRuntime"
        xticklabels[-2] = f"1.0\n1 - Dice"
        ax.set_xticks(xticks[1:-1])
        ax.set_xticklabels(xticklabels[1:-1])

        fig.subplots_adjust(
            top=0.86,
            bottom=0.34,
            left=0.18,
            right=0.95,
            wspace=0.2,
            hspace=0.5,
        )

        return fig


    def _plot_ablation_path(
        self,
        dataset: str, 
        approach_key: str,
        budget_idx: int = COMBINED_BUDGET,
        n_hps: int = 3
    ):
        try:
            if approach_key == "hpo":
                deepcave_run = self.get_hpo_data(dataset).deepcave_runs[dataset]
            elif approach_key == "hpo_nas":
                deepcave_run = self.get_nas_data(dataset).deepcave_runs[dataset]
            elif approach_key == "hpo_hnas":
                deepcave_run = self.get_hnas_data(dataset).deepcave_runs[dataset]
            else:
                raise ValueError(f"Unknown approach key {approach_key}.")
        except KeyError:
            return
        
        selected_budget = self._get_budget(budget_idx, deepcave_run)
        
        if approach_key == "hpo":
            evaluator = Ablation(run=deepcave_run)
            evaluator.calculate(
                objectives=deepcave_run.get_objectives(),
                budget=selected_budget,
                seed=42
            )
            
            performances = evaluator.get_ablation_performances()
            improvements = evaluator.get_ablation_improvements()

            assert performances is not None
            assert improvements is not None

            # We need to check if all improvements are zero, 
            # as this would mean that the ablation path is flat
            _improvement = 0
            for key, value in improvements.items():
                _improvement += value[0]
            if _improvement == 0:
                return

            # Get first n_hps elements (including the baseline)
            performances = dict(list(performances.items())[:n_hps + 1])
            improvements = dict(list(improvements.items())[:n_hps + 1])

            fig = self._plot_single_objective_ablation(
                performances=performances,
                improvements=improvements
            )
        else:
            evaluator = MOAblation(run=deepcave_run)
            evaluator.calculate(
                objectives=deepcave_run.get_objectives(),
                budget=selected_budget,
                seed=42
            )  

            data = evaluator.df_importances
            if data["importance"].eq(0).all():
                return
            
            idx = (
                data
                .groupby("hp_name")["importance"]
                .max()
                .sort_values(ascending=False)
                .index
            )
            idx = list(idx[:n_hps]) + ["Default"]
            importances = data[
                data["hp_name"].isin(idx)
            ].copy()

            fig = self._plot_multi_objective_ablation(
                approach_key=approach_key,
                importances=importances,
            )
        _approach = APPROACH_REPLACE_MAP[approach_key].replace(' (ours)', '')
        if approach_key == "hpo":
            fig.suptitle(f"Ablation Path for {_approach} on {format_dataset_name(dataset)}")
        else:
            fig.suptitle(f"MO Ablation Path for {_approach} on\n{format_dataset_name(dataset)}")
        plt.savefig(
            self.analysis_plots[approach_key] / f"ablation_{dataset}.{self.format}",
            format=self.format, 
            dpi=self.dpi
        )

    def plot_ablation_paths(
        self,
        approach_keys: list[str] | None = None,
        budget_idx: int = 4,
        n_hps: int = 3
    ):
        if approach_keys is None:
            approach_keys = APPROACHES
        for approach_key in approach_keys:
            for dataset in self.datasets:
                self._plot_ablation_path(
                    dataset=dataset,
                    approach_key=approach_key,
                    budget_idx=budget_idx,
                    n_hps=n_hps
                )

    def _plot_pdp_1hp(self, deepcave_run: DeepCAVERun, outputs: dict, show_ice: bool, hp_name: str):
        x = np.asarray(outputs["x"])
        y = np.asarray(outputs["y"])
        sigmas = np.sqrt(np.asarray(outputs["variances"]))
        x_ice = np.asarray(outputs["x_ice"])
        y_ice = np.asarray(outputs["y_ice"])

        hp_idx = deepcave_run.configspace.index_of[hp_name]
        hp = deepcave_run.configspace[hp_name]

        fig, ax = plt.subplots(1, 1, figsize=(self.figwidth / 2, 3))

        if show_ice:
            for x_, y_ in zip(x_ice, y_ice):
                sns.lineplot(
                    x=x_[:, hp_idx],
                    y=y_,
                    ax=ax,
                    color=self.color_palette[3],
                    alpha=0.1,

                )
        
        sns.lineplot(
            x=x[:, hp_idx],
            y=y,
            ax=ax,
            color=self.color_palette[0],
            linewidth=2
        )

        ax.fill_between(
            x[:, hp_idx],
            y - sigmas,
            y + sigmas,
            color=self.color_palette[0],
            alpha=0.2
        )

        tickvals, ticktext = get_hyperparameter_ticks(hp)
        ax.set_xticks(tickvals[:-1])
        ax.set_xticklabels(ticktext[:-1])
        ax.set_xlabel(HYPERPARAMETER_REPLACEMENT_MAP[hp_name])
        ax.set_ylabel("1 - Dice")

        fig.subplots_adjust(
            top=0.91,
            bottom=0.15,
            left=0.15,
            right=0.98,
            wspace=0.33,
            hspace=0.33
        )

        return fig
    
    def _plot_pdp_2hps(
            self,
            deepcave_run: DeepCAVERun,
            outputs: dict,
            hp_name_1: str,
            hp_name_2: str,
        ):
        x = np.asarray(outputs["x"])
        y = np.asarray(outputs["y"])
        sigmas = np.sqrt(np.asarray(outputs["variances"]))
        x_ice = np.asarray(outputs["x_ice"])
        y_ice = np.asarray(outputs["y_ice"])

        hp1_idx = deepcave_run.configspace.index_of[hp_name_1]
        hp1 = deepcave_run.configspace[hp_name_1]
        hp2_idx = deepcave_run.configspace.index_of[hp_name_2]
        hp2 = deepcave_run.configspace[hp_name_2]

        fig, ax = plt.subplots(1, 1, figsize=(self.figwidth / 2, 3))

        x_hp1 = x[:, hp1_idx]
        x_hp2 = x[:, hp2_idx]

        contour = plt.tricontourf(x_hp1, x_hp2, y, levels=15, cmap="plasma", alpha=1)  # Create filled contours
        cbar = plt.colorbar(contour)
        cbar.set_label("1 - Dice")

        xtickvals, xticktext = get_hyperparameter_ticks(hp1)
        ytickvals, yticktext = get_hyperparameter_ticks(hp2)
        ax.set_xticks(xtickvals[:-1])
        ax.set_xticklabels(xticktext[:-1])
        ax.set_yticks(ytickvals[:-1])
        ax.set_yticklabels(yticktext[:-1])
        ax.set_xlabel(HYPERPARAMETER_REPLACEMENT_MAP[hp_name_1])
        ax.set_ylabel(HYPERPARAMETER_REPLACEMENT_MAP[hp_name_2])

        fig.subplots_adjust(
            top=0.89,
            bottom=0.15,
            left=0.22,
            right=0.92,
        )

        return fig

    def plot_pdp(
            self,
            dataset: str,
            approach_key: str,
            hp_name_1: str,
            hp_name_2: str | None = None,
            budget_id: int = COMBINED_BUDGET,
            objective_id: int = 0,
            show_ice: bool = True
    ):
        try:
            if approach_key == "hpo":
                deepcave_run = self.get_hpo_data(dataset).deepcave_runs[dataset]
            elif approach_key == "hpo_nas":
                deepcave_run = self.get_nas_data(dataset).deepcave_runs[dataset]
            elif approach_key == "hpo_hnas":
                deepcave_run = self.get_hnas_data(dataset).deepcave_runs[dataset]
            else:
                raise ValueError(f"Unknown approach key {approach_key}.")
        except Exception as e:
            print(e)
            return
        
        inputs = {
            "objective_id": objective_id,
            "budget_id": budget_id,
            "hyperparameter_name_1": hp_name_1,
            "hyperparameter_name_2": hp_name_2,
        }

        pdp = PartialDependencies()
        outputs = pdp.process(
            run=deepcave_run,
            inputs=inputs
        )

        if hp_name_2 is None:
            fig = self._plot_pdp_1hp(
                deepcave_run=deepcave_run,
                outputs=outputs,
                show_ice=show_ice,
                hp_name=hp_name_1
            )
        else:
            fig = self._plot_pdp_2hps(
                deepcave_run=deepcave_run,
                outputs=outputs,
                hp_name_1=hp_name_1,
                hp_name_2=hp_name_2
            )
        
        plt.title(f"PDP for {APPROACH_REPLACE_MAP[approach_key].replace(' (ours)', '')} on {format_dataset_name(dataset)}")

        if hp_name_2 is None:
            title = f"pdp_{dataset}_{hp_name_1}"
        else:
            title = f"pdp_{dataset}_{hp_name_1}_{hp_name_2}"

        plt.savefig(
            self.analysis_plots[approach_key] / f"{title}.{self.format}", 
            format=self.format, 
            dpi=self.dpi
        )

    def _plot_footprint(
            self,
            dataset: str,
            approach_key: str,
            objective: str = "1 - Dice",
            budget: int = COMBINED_BUDGET
        ):
        try:
            deepcave_run, history, incumbent = self.get_deepcave_data(dataset, approach_key, objective)
        except KeyError:
            return 
        
        fp = Footprint(
            run=deepcave_run
        )

        selected_budget = self._get_budget(budget, deepcave_run)

        _objective = deepcave_run.get_objective(objective)
        assert _objective is not None
        
        cache_dir = DEEPCAVE_CACHE_DIR / approach_key / dataset
        if (cache_dir / f"footprint_{objective}_{budget}_configs_x.npy").exists():
            configs_x = np.load(cache_dir / f"footprint_{objective}_{budget}_configs_x.npy")
            configs_y = np.load(cache_dir / f"footprint_{objective}_{budget}_configs_y.npy")
            config_ids = np.load(cache_dir / f"footprint_{objective}_{budget}_config_ids.npy")

            surface_x = np.load(cache_dir / f"footprint_{objective}_{budget}_surface_x.npy")
            surface_y = np.load(cache_dir / f"footprint_{objective}_{budget}_surface_y.npy")
            surface_z = np.load(cache_dir / f"footprint_{objective}_{budget}_surface_z.npy")
        else:
            fp.calculate(
                objective=_objective,
                budget=selected_budget,
            )

            configs_x, configs_y, config_ids = fp.get_points()
            surface_x, surface_y, surface_z = fp.get_surface()

            cache_dir.mkdir(parents=True, exist_ok=True)

            np.save(cache_dir / f"footprint_{objective}_{budget}_configs_x.npy", configs_x)
            np.save(cache_dir / f"footprint_{objective}_{budget}_configs_y.npy", configs_y)
            np.save(cache_dir / f"footprint_{objective}_{budget}_config_ids.npy", config_ids)

            np.save(cache_dir / f"footprint_{objective}_{budget}_surface_x.npy", surface_x)
            np.save(cache_dir / f"footprint_{objective}_{budget}_surface_y.npy", surface_y)
            np.save(cache_dir / f"footprint_{objective}_{budget}_surface_z.npy", surface_z)

        z_min, z_max = np.min(surface_z), np.max(surface_z)

        z_min = np.floor(z_min * 10) / 10
        z_max = np.ceil(z_max * 10) / 10

        tick_step = 0.1 
        ticks = np.arange(z_min, z_max + tick_step, tick_step)

        fig, ax = plt.subplots(1, 1, figsize=(self.figwidth / 2, self.figwidth / 2))
        cmap = sns.dark_palette(self.color_palette[0], reverse=True, as_cmap=True)
        heatmap = plt.contourf(surface_x, surface_y, surface_z, levels=100, cmap=cmap)
        cbar = plt.colorbar(heatmap, ticks=ticks)
        cbar.set_label(objective, labelpad=10)
        
        origins = ["Default", "Incumbent"] + list(ORIGIN_MAP.values()) 
        inc_config_id = incumbent["Configuration ID"].values[-1]

        history_unique = history.drop_duplicates(subset="Configuration ID")

        config_origin = dict(zip(history_unique["Configuration ID"], history_unique["Config Origin"]))
        config_origin[inc_config_id] = "Incumbent"

        config_marker = {origin: marker for origin, marker in zip(origins, ["v", "^", "x", "x", "x"])}
        
        # We remove light blue from the palette here
        color_palette = self.color_palette[:5] + [self.color_palette[8]]
        config_color = {origin: color for origin, color in zip(origins, color_palette[1:len(origins) + 1])}

        config_size = history["Configuration ID"].value_counts().to_dict()
        config_size = {k: v / max(config_size.values()) for k, v in config_size.items()}
        config_size[inc_config_id] = 1.25
        config_size[0] = 1.25    # Default

        config_zorder = {
            "Rand. Sampling": 1,
            "Prior Sampling": 2,
            "Inc. Sampling": 3,
            "Default": 4,
            "Incumbent": 5,
        }

        legend_elements = []

        for x, y, id in zip(configs_x, configs_y, config_ids):
            id = int(id)
            origin = config_origin[id]
            ax.scatter(
                x=x,
                y=y,
                label=config_origin[id],
                marker=config_marker[origin],
                color=config_color[origin],
                s=60 * config_size[id],
                zorder=config_zorder[origin],
                edgecolor="black" if origin in ["Default", "Incumbent"] else None,
                linewidth=1 if origin in ["Default", "Incumbent"] else None,
            )

        legend_elements += [
            Line2D(
                [0],
                [0],
                marker=config_marker[origin],
                color=config_color[origin],
                label=origin,
                markerfacecolor=config_color[origin],
                markersize=8,
                linestyle="None",
                markeredgecolor="black" if origin in ["Default", "Incumbent"] else None,
                linewidth=1 if origin in ["Default", "Incumbent"] else None,
            )
            for origin in origins
        ]

        # We add a dummy to move the sampling strategies to the right
        dummy_element = Line2D([0], [0], color="none", label="")
        legend_elements.insert(2, dummy_element) 

        ax.legend(
            handles=legend_elements,
            loc="upper center", 
            bbox_to_anchor=(0.6, -0.03),
            ncol=2,
            title="Configuration Type      Sampling Strategy  ",
            fancybox=False,
            shadow=False,
            frameon=False
        )

        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")
        plt.title(f"Configuration Footprint for\n{APPROACH_REPLACE_MAP[approach_key].replace(' (ours)', '')} on {format_dataset_name(dataset)}")

        fig.subplots_adjust(
            top=0.89,
            bottom=0.25,
            left=0.03,
            right=0.93,
        )

        plt.savefig(
            self.analysis_plots[approach_key] / f"footprint_{dataset}.{self.format}",
            format=self.format,
            dpi=self.dpi
        )
        plt.clf()

    def plot_footprints(
            self,
            approach_keys: list[str] | None = None,
            budget: int | Literal["combined"] = "combined"
        ):
        if approach_keys is None:
            approach_keys = APPROACHES
        for approach_key in ["hpo", "hpo_nas", "hpo_hnas"]:
            for dataset in self.datasets:
                self._plot_footprint(dataset=dataset, approach_key=approach_key, budget=budget)

    def _get_corr_categories(
            self,
            correlations: dict
        ) -> dict:
        categories = {}

        for budget1, budgets in correlations.items():
            for budget2, correlation in budgets.items():
                if budget1 == budget2:
                    continue
                
                if correlation >= 0.7:
                    category = "Very strong"
                elif correlation >= 0.4:
                    category = "Strong"
                elif correlation >= 0.3:
                    category = "Moderate"
                elif correlation >= 0.2:
                    category = "Weak"
                else:
                    category = "Not given"

                budget1 = round(budget1)
                budget2 = round(budget2)

                key = (budget1, budget2)
                key2 = (budget2, budget1)
                if float(budget1) < float(budget2):
                    categories[key2] = category
                else:
                    categories[key] = category

        return categories

    def _plot_budget_correlations(
            self,
            dataset: str,
            approach_key: str,
            objective: str = "1 - Dice",
        ):
        def round_dict_keys(d: dict) -> dict:
            """Helper for rounding budget  keys."""
            new_dict = {}
            for k, v in d.items():
                try:
                    k = round(float(k))
                except ValueError:
                    pass
                    
                if isinstance(v, dict):
                    v = round_dict_keys(v)
                
                new_dict[k] = v
            return new_dict
        
        try:
            deepcave_run, _, _ = self.get_deepcave_data(dataset, approach_key, objective)
        except KeyError:
            return
        
        objective_id = deepcave_run.get_objective_id(objective)
        assert objective_id is not None

        budget_corr = BudgetCorrelation.process(
            run=deepcave_run,
            inputs={"objective_id": objective_id}
        )

        correlations_symmetric = budget_corr["correlations_symmetric"]
        correlations = budget_corr["correlations"]
        correlations_symmetric = round_dict_keys(correlations_symmetric)
        correlations = round_dict_keys(correlations)
        
        categories = self._get_corr_categories(correlations=correlations)

        fig, ax = plt.subplots(1, 1, figsize=(self.figwidth, self.figwidth / 2))
        budgets = deepcave_run.get_budgets(include_combined=False)
        budgets = [round(b) for b in budgets]
        palette = self.color_palette[:len(budgets)]
        markers = {
            "Very strong": "o",
            "Strong": "s",
            "Moderate": "D",
            "Weak": "P",
            "Not given": "X"
        }

        # First, we add vertical lines for each budget 
        for idx, budget in enumerate(budgets):
            plt.axvline(
                x=budget,
                color=self.color_palette[idx],
                linestyle="--",
                linewidth=1,
                label=f"Budget {budget}",
            )

        for i, b1 in enumerate(budgets):
            budget_x = []
            budget_y = []
            for j, b2 in enumerate(budgets):
                if b2 >= b1:
                    continue

                corr = correlations_symmetric[b1][b2]

                if np.isnan(corr):
                    continue

                category = categories[(b1, b2)]
                marker = markers[category]

                sns.scatterplot(
                    x=[b2],
                    y=[corr],
                    label=f"Budget {b1}",
                    color=palette[i],
                    marker=marker,
                    s=50,
                    zorder=2
                )

                budget_x += [b2]
                budget_y += [corr]

            sns.scatterplot(
                x=[b1],
                y=[1],
                label=f"Budget {b1}",
                color=palette[i],
                marker="o",
                s=50,
                zorder=3
            )

            budget_x += [b1]
            budget_y += [1]

            sns.lineplot(
                x=budget_x,
                y=budget_y,
                color=palette[i],
                linewidth=1,
            )

        # No we manually need to create the legend for both marker types and budgets
        marker_legend_elements = []
        for category, marker in markers.items():
            marker_legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="w",
                    label=category,
                    markerfacecolor="k",
                    markersize=10,
                    linestyle="None",
                )
            )
        
        budget_legend_elements = []
        for idx, budget in enumerate(budgets):
            budget_legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=self.color_palette[idx],
                    label=f"{budget}",
                    linestyle="-",
                    linewidth=1,
                )
            )

        # First legend: Correlation Category
        legend1 = ax.legend(
            handles=marker_legend_elements,
            title="Correlation Category",
            loc="upper center",
            bbox_to_anchor=(0.25, -0.15),
            ncol=2,
            fancybox=False,
            shadow=False,
            frameon=False
        )

        # Add the first legend to the axes
        ax.add_artist(legend1)

        # Second legend: Budget
        legend2 = ax.legend(
            handles=budget_legend_elements,
            title="Budget",
            loc="upper center",
            bbox_to_anchor=(0.75, -0.15),
            ncol=2,
            fancybox=False,
            shadow=False,
            frameon=False
        )

        # Customize the plot
        plt.xlabel("Budget")
        plt.ylabel("Correlation")
        plt.title(f"Budget Correlation for {APPROACH_REPLACE_MAP[approach_key].replace(' (ours)', '')} on {format_dataset_name(dataset)}")
        plt.grid(True)

        fig.subplots_adjust(
            top=0.92,
            bottom=0.34,
            left=0.09,
            right=0.98,
        )

        plt.savefig(
            self.analysis_plots[approach_key] / f"budget_correlation_{dataset}.{self.format}",
            format=self.format,
            dpi=self.dpi
        )
        plt.clf()

    def plot_budget_correlations(
            self,
            approach_keys: list[str] | None = None
        ):
        if approach_keys is None:
            approach_keys = APPROACHES
        for approach_key in approach_keys:
            for dataset in self.datasets:
                self._plot_budget_correlations(dataset=dataset, approach_key=approach_key)

    def create_emissions_table(self):
        self.load_all_data()
        baseline_emissions = self._baseline_data.emissions[["run_id", "Approach", "Fold", "Dataset", "emissions"]]
        hpo_emissions = self._hpo_data.emissions[["run_id", "Approach", "Fold", "Dataset", "emissions"]]
        nas_emissions = self._nas_data.emissions[["run_id", "Approach", "Fold", "Dataset", "emissions"]]
        hnas_emissions = self._hnas_data.emissions[["run_id", "Approach", "Fold", "Dataset", "emissions"]]

        emissions = pd.concat([baseline_emissions, hpo_emissions, nas_emissions, hnas_emissions])
        emissions["Dataset"] = emissions["Dataset"].apply(format_dataset_name)

        # We format the approach names to make the table smaller. 
        # Also, we add an index we use to sort based on the approach
        emissions.loc[:, "approach_idx"] = 0
        for approach in emissions["Approach"].unique():
            if "nnU-Net" in approach:
                new_approach = "Baseline"
                idx = 0
            else:
                new_approach = approach.replace(" (ours)", "")
                if new_approach == "HPO":
                    idx = 1
                elif new_approach == "HPO + NAS":
                    idx = 2
                else: 
                    idx = 3

            emissions.loc[emissions["Approach"] == approach, "Approach"] = new_approach
            emissions.loc[emissions["Approach"] == approach, "approach_idx"] = idx
        
        emissions = emissions.sort_values(by=["approach_idx", "Dataset", "Fold"])
        emissions = emissions.drop(columns=["approach_idx", "Fold", "run_id"])

        emissions_per_dataset = emissions.groupby(["Dataset", "Approach"])["emissions"].sum().reset_index()
        
        overall_sum = emissions.groupby("Approach")["emissions"].sum().reset_index()
        overall_sum["Dataset"] = "Overall"  
        
        emissions_combined = pd.concat([emissions_per_dataset, overall_sum], ignore_index=True)
        emissions_table = emissions_combined.pivot(index="Dataset", columns="Approach", values="emissions").fillna(0)

        emissions_table.to_latex(
            AUTONNUNET_TABLES / "emissions_table.tex",
            float_format="%.2f"
        )


    def create_table(self, datasets: list[str]):
        baseline = self._load_baseline_data(datasets=datasets)
        hpo = self._load_hpo_data(datasets=datasets)

        sys.exit()

        # We remove all background classes, since we compute the mean foreground Dice
        baseline.metrics = baseline.metrics[baseline.metrics["class_id"] != 0]
        hpo.metrics = hpo.metrics[hpo.metrics["class_id"] != 0]

        # We can drop all other metrics
        baseline.metrics = baseline.metrics[["Dataset", "Fold", "Approach", "Dice"]]
        hpo.metrics = hpo.metrics[["Dataset", "Fold", "Approach", "Dice"]]

        # Then, we average over all prediction files
        baseline.metrics = baseline.metrics.groupby(
            ["Dataset", "Fold", "Approach"]).mean().reset_index()
        hpo.metrics = hpo.metrics.groupby(
            ["Dataset", "Fold", "Approach"]).mean().reset_index()

        metrics = pd.concat([baseline.metrics, hpo.metrics])
        metrics["Dataset"] = metrics["Dataset"].apply(format_dataset_name)

        grouped_metrics = metrics.groupby(["Dataset", "Approach"])["Dice"].agg(
            ["mean", "std"]).reset_index()

        # First, we add the mean and standard deviation
        grouped_metrics["mean"] = grouped_metrics["mean"].round(2)
        grouped_metrics["std"] = grouped_metrics["std"].round(2)
        grouped_metrics["mean_std"] = grouped_metrics["mean"].astype(str) \
            + " $\\pm$ " + grouped_metrics["std"].astype(str)

        # We highlight the best approach per dataset
        max_vals = grouped_metrics.groupby("Dataset")["mean"].transform("max")
        max_mean = max_vals == grouped_metrics["mean"]
        grouped_metrics["mean_std"] = grouped_metrics.apply(
            lambda row: "\\textbf{" + row["mean_std"] + "}"
            if max_mean.loc[row.name]
            else row["mean_std"], axis=1
        )

        table = grouped_metrics.pivot_table(
            index="Dataset",
            columns="Approach",
            values="mean_std"
        )
        table.to_latex(AUTONNUNET_TABLES / f"{self.configuration}_table.tex")

    def plot_baseline_runtimes(self):
        baseline_runtimes = self._baseline_data.progress[["Approach", "Dataset", "Fold", "Runtime"]]
        dataset_runtimes = baseline_runtimes.groupby(["Approach", "Dataset", "Fold"]).sum().reset_index()
        dataset_runtimes["Runtime"] /= 3600

        dataset_runtimes["Dataset"] = dataset_runtimes["Dataset"].apply(lambda s: format_dataset_name(s).replace(" ", "\n"))

        # Create bar plot of dataset runtimes and mark average
        # Use different colors for each bar
        plt.figure(1, figsize=(self.figwidth, self.figwidth / 2))
        g = sns.barplot(
            x="Dataset",
            y="Runtime",
            hue="Approach",
            hue_order=list(APPROACH_REPLACE_MAP.values())[:3],
            data=dataset_runtimes,
        )

        # Plot mean per approach
        current_approach = 0
        for approach in list(APPROACH_REPLACE_MAP.values())[:3]:
            avg_runtime = dataset_runtimes[dataset_runtimes["Approach"] == approach]["Runtime"].mean()
            plt.axhline(y=avg_runtime, color=self.color_palette[current_approach], linestyle="--")
            current_approach += 1

        plt.title("Runtime per Approach and Dataset")
        g.set_xlabel("Dataset")
        g.set_ylabel("Runtime [h]")

        plt.tight_layout()
        plt.savefig(
            self.baseline_plots / f"{self.configuration}_runtimes.{self.format}",
            format=self.format,
            dpi=self.dpi
        )
        plt.clf()

    def plot_baseline_performances_and_runtimes(self):
        # Create bar plot of dataset runtimes and mark average
        # Use different colors for each bar
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            figsize=(self.figwidth, 4.5)
        )

        # Plot performances
        baseline_metrics = self._baseline_data.metrics_foreground_mean
        baseline_metrics["Dataset"] = baseline_metrics["Dataset"].apply(lambda s: format_dataset_name(s).replace(" ", "\n"))

        g = sns.barplot(
            x="Dataset",
            y="Dice",
            hue="Approach",
            hue_order=list(APPROACH_REPLACE_MAP.values())[:3],
            data=baseline_metrics,
            ax=axes[0]
        )
        axes[0].set_title("Baseline Performances")
        axes[0].set_ylim(0, 1)

        baseline_runtimes = self._baseline_data.progress[["Approach", "Dataset", "Fold", "Runtime"]]
        dataset_runtimes = baseline_runtimes.groupby(["Approach", "Dataset", "Fold"]).sum().reset_index()
        dataset_runtimes["Runtime"] /= 3600

        dataset_runtimes["Dataset"] = dataset_runtimes["Dataset"].apply(lambda s: format_dataset_name(s).replace(" ", "\n"))

        g = sns.barplot(
            x="Dataset",
            y="Runtime",
            hue="Approach",
            hue_order=list(APPROACH_REPLACE_MAP.values())[:3],
            data=dataset_runtimes,
            ax=axes[1]
        )
        axes[1].set_title("Baseline Runtimes")

        # Plot mean runtime per approach
        current_approach = 0
        for approach in list(APPROACH_REPLACE_MAP.values())[:3]:
            avg_score = baseline_metrics[baseline_metrics["Approach"] == approach]["Dice"].mean()
            axes[0].axhline(y=avg_score, color=self.color_palette[current_approach], linestyle="--")
            avg_runtime = dataset_runtimes[dataset_runtimes["Approach"] == approach]["Runtime"].mean()
            axes[1].axhline(y=avg_runtime, color=self.color_palette[current_approach], linestyle="--")
            current_approach += 1

        for ax in axes:
            ax.legend().remove()

        g.set_xlabel("Dataset")
        g.set_ylabel("Runtime [h]")

        axes[1].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.27),
            ncol=3,
            fancybox=False,
            shadow=False,
            frameon=False
        )

        plt.tight_layout()
        plt.savefig(
            self.baseline_plots / f"{self.configuration}_performances_and_runtimes.{self.format}",
            format=self.format,
            dpi=self.dpi
        )
        plt.clf()


    def _read_msd_results(self, approaches: list[str]) -> pd.DataFrame:
        results = []

        for approach in approaches:
            result_path = AUTONNUNET_MSD_RESULTS / f"{approach}_{self.configuration}.json"
            if not result_path.exists():
                continue

            msd_results = load_json(result_path)

            for task_id, task_results in msd_results.items():
                dataset = msd_task_to_dataset_name(task_id)

                for cls in task_results["aggregates"]:
                    results += [{
                        "Dataset": dataset,
                        "Class": cls,
                        "Approach": APPROACH_REPLACE_MAP[approach],
                        **task_results["aggregates"][cls]
                    }]

        return pd.DataFrame(results)

    def compare_msd_results(self):
        approaches = ["baseline_ConvolutionalEncoder", "hpo", "hpo_nas"]
        msd_results = self._read_msd_results(approaches)


        msd_results_per_dataset = msd_results[["Approach", "Dataset", "mean"]].groupby(["Approach", "Dataset"]).mean().reset_index()
        msd_results_per_dataset = msd_results_per_dataset.rename(columns={"mean": "Dice"})

        print(msd_results_per_dataset)

    def plot_msd_overview(self):
        def get_slice(img, label):
            img_data = img.get_fdata()
            label_data = label.get_fdata()

            if len (img_data.shape) == 4:
                img_data = img_data[:, :, :, 0]

            max_foreground = 0
            slice_idx = 0
            for i in range(label_data.shape[2]):
                if (fg_sum := label_data[:, :, i].sum()) > max_foreground:
                    max_foreground = fg_sum
                    slice_idx = i

            img_slice = img_data[:, :, slice_idx]
            label_slice = label_data[:, :, slice_idx]

            return img_slice, label_slice

        def get_slices(dataset: str) -> tuple[np.ndarray, np.ndarray]:
            dataset_path = NNUNET_DATASETS / dataset_name_to_msd_task(dataset)

            for file in (dataset_path / "imagesTr" ).glob("*.nii.gz"):
                if file.name.startswith("._"):
                    continue

                img = nib.loadsave.load(file)
                label = nib.loadsave.load(dataset_path / "labelsTr" / file.name)

                img_slice, label_slice = get_slice(img, label)
                label_slice[label_slice == 0] = np.nan

                # We just use the first image
                break

            return img_slice, label_slice

        fig, axs = plt.subplots(
            ncols=5,
            nrows=2,
            figsize=(8, 4.25),
        )

        output_dir = AUTONNUNET_PLOTS / "related_work"
        output_dir.mkdir(parents=True, exist_ok=True)

        images = []
        labels = []

        for dataset, ax in zip(self.datasets, axs.flatten(), strict=False):
            img, label = get_slices(dataset)

            images.append(img)
            labels.append(label)

            ax.set_title(format_dataset_name(dataset).replace(" ", "\n"))
            ax.imshow(img, cmap="gray")
            ax.imshow(label, cmap="viridis", alpha=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / "msd_overview.png", dpi=500)
        plt.clf()

        for dataset, img, label in zip(self.datasets, images, labels, strict=False):
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))

            ax.imshow(img, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")

            plt.tight_layout()
            plt.savefig(output_dir / f"{dataset}_img.png", dpi=500)
            plt.clf()

            fig, ax = plt.subplots(1, 1, figsize=(4, 4))

            ax.imshow(img, cmap="gray")
            ax.imshow(label, cmap="viridis", alpha=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")

            plt.tight_layout()
            plt.savefig(output_dir / f"{dataset}_label.png", dpi=500)
            plt.close()