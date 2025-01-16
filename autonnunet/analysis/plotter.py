from __future__ import annotations

import ast
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from deepcave.evaluators.fanova import fANOVA
from deepcave.evaluators.footprint import Footprint

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

HYPERPARAMETER_REPLACEMENT_MAP = {
    "optimizer": "Optimizer",
    "momentum": "Momentum (SGD)",
    "initial_lr": "Initial LR",
    "lr_scheduler": "LR Scheduler",
    "weight_decay": "Weight Decay",
    "loss_function": "Loss Function",
    "aug_factor": "Data Augmentation [%]",
    "oversample_foreground_percent": "Oversample Foreground [%]",
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

PROGRESS_FILENAME = "progress.csv"
HISTORY_FILENAME = "runhistory.csv"
SAMPLING_POLICY_LOGS = "sampling_policy.log"
INCUMBENT_FILENAME = "incumbent_loss.csv"
VALIDATION_METRICS_FILENAME = "summary.json"
EMISSIONS_FILENAME = "emissions.csv"
DATASET_JSON_FILENAME = "dataset.json"

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
            configuration: str,
            datasets: list[str],
            objectives: list[str] | None = None,
            min_budget: float = 10.0,
            max_budget: float = 1000.0,
            eta: int = 3,
            n_folds: int = 5,
            style: STYLES_TYPE = "whitegrid",
            palette: str = "colorblind",
            figwidth: int = 8,
            hpo_seed: int = 0,
        ):
        if objectives is None:
            objectives = ["1 - Dice", "Runtime"]
        self.datasets = datasets
        self.objectives = objectives

        # We need these to find the respective directories
        self.configuration = configuration

        self.min_budget = min_budget
        self.max_budget = max_budget

        self.n_folds = n_folds
        self.hpo_seed = hpo_seed

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

        # Directories
        self.baseline_conv =  AUTONNUNET_OUTPUT / "baseline_ConvolutionalEncoder"
        self.baseline_resenc_m =  AUTONNUNET_OUTPUT / "baseline_ResidualEncoderM"
        self.baseline_resenc_l =  AUTONNUNET_OUTPUT / "baseline_ResidualEncoderL"
        self.hpo_dir = AUTONNUNET_OUTPUT / "hpo"
        self.nas_dir = AUTONNUNET_OUTPUT / "hpo_nas"
        self.hnas_dir = AUTONNUNET_OUTPUT / "hpo_hnas"

        # Seaborn settings
        sns.set_style(style=style)
        sns.set_palette(palette=palette)
        self.palette = palette

        # Matplotlib settings
        self.figwidth = figwidth

        self.logger = logging.getLogger("Plotter")

        self.hpo_plots = AUTONNUNET_PLOTS / "hpo"
        self.hpo_plots.mkdir(parents=True, exist_ok=True)
        self.nas_plots = AUTONNUNET_PLOTS / "hpo_nas"
        self.nas_plots.mkdir(parents=True, exist_ok=True)
        self.hnas_plots = AUTONNUNET_PLOTS / "hpo_hnas"
        self.hnas_plots.mkdir(parents=True, exist_ok=True)
        self.baseline_plots = AUTONNUNET_PLOTS / "baseline"
        self.baseline_plots.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        self._baseline_data = self._load_baseline_data(datasets=self.datasets)
        self._hpo_data = self._load_hpo_data(datasets=self.datasets)
        self._nas_data = self._load_nas_data(datasets=self.datasets, approach_key="hpo_nas")
        self._hnas_data = self._load_nas_data(datasets=self.datasets, approach_key="hpo_hnas")

        self._baseline_datasets = self._baseline_data.progress["Dataset"].unique().tolist()
        self._hpo_datasets = self._hpo_data.history["Dataset"].unique().tolist()
        self._nas_datasets = self._nas_data.history["Dataset"].unique().tolist()
        self._hnas_datasets = self._nas_data.history["Dataset"].unique().tolist()

        self.logger.info(
            f"Loaded {len(self._baseline_datasets)} datasets for baseline.")
        self.logger.info(
            f"Loaded {len(self._hpo_datasets)} datasets for HPO.")
        self.logger.info(
            f"Loaded {len(self._nas_datasets)} datasets for HPO + NAS.")
        self.logger.info(
            f"Loaded {len(self._hnas_datasets)} datasets for HPO + HNAS.")

    def get_baseline_data(self, dataset: str):
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
        prior_band_incumbent_progress = self._hpo_data.incumbent_progress[
            self._hpo_data.incumbent_progress["Dataset"] == dataset]
        prior_band_emissions = self._hpo_data.emissions[
            self._hpo_data.emissions["Dataset"] == dataset]
        prior_band_history = self._hpo_data.history[
            self._hpo_data.history["Dataset"] == dataset]
        prior_band_incumbent = self._hpo_data.incumbent[
            self._hpo_data.incumbent["Dataset"] == dataset]

        return HPOResult(
            incumbent_progress=prior_band_incumbent_progress,
            emissions=prior_band_emissions,
            history=prior_band_history,
            incumbent=prior_band_incumbent,
            deepcave_runs=self._hpo_data.deepcave_runs
        )

    def get_nas_data(self, dataset: str):
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
            deepcave_runs=self._nas_data.deepcave_runs
        )
    
    def get_hnas_data(self, dataset: str):
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
            deepcave_runs=self._hnas_data.deepcave_runs
        )

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

        origins = {i + 1: origin for i, origin in enumerate(lines)}
        origins[0] = "default"

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

    def _load_nas_data(self, datasets: list[str], approach_key: str) -> NASResult:
        all_emissions = []
        all_history = []
        all_incumbent = defaultdict(list)
        deepcave_runs = {}
        
        if approach_key == "hpo_nas":
            base_dir = self.nas_dir 
        elif approach_key == "hpo_hnas":
            base_dir = self.hnas_dir
        else:
            raise ValueError(f"Unknown approach key {approach_key}.")

        approach = APPROACH_REPLACE_MAP[approach_key]

        for dataset in datasets:
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
                approach=approach_key
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

    def _load_hpo_data(self, datasets: list[str]) -> HPOResult:
        all_progress  = []
        all_emissions = []
        all_history = []
        all_incumbent = []
        deepcave_runs = {}

        approach_key = "hpo"
        approach = APPROACH_REPLACE_MAP[approach_key]

        for dataset in datasets:
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
                approach=approach_key
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

    def _plot_baseline(self, dataset: str) -> None:
        baseline_data = self.get_baseline_data(dataset)

        plt.figure(1, figsize=(self.figwidth, 4.5))

        g = sns.lineplot(
            x="Epoch",
            y="mean_fg_dice",
            data=baseline_data.progress,
            label="Mean Foreground Dice",
            errorbar=("sd")
        )

        g.set_title(f"Training Progress for {format_dataset_name(dataset)}")
        g.set_xlabel("Epoch")
        g.set_ylabel("Mean Foreground Dice")

        plt.minorticks_on()
        plt.grid(visible=True, which="both", linestyle="--")
        plt.grid(visible=True, which="minor", linestyle=":", linewidth=0.5)

        plt.tight_layout()
        plt.savefig(
            self.baseline_plots / f"{dataset}_{self.configuration}.png",
            dpi=400
        )

        plt.clf()

    def plot_baselines(
            self,
        ):
        for dataset in self._baseline_datasets:
            self._plot_baseline(dataset)

    def _plot_hpo_(
            self,
            dataset: str,
            x_log_scale: bool = False,
            y_log_scale: bool = False,
            include_nas: bool = False,
            include_hnas: bool = False,
            show_error: bool = False
        ) -> None:
        plt.figure(1, figsize=(5.5, 4.5))

        hpo_data = self.get_hpo_data(dataset)
        incumbent = hpo_data.incumbent

        if include_nas and dataset in self._nas_datasets:
            nas_data = self.get_nas_data(dataset)
            incumbent = pd.concat([incumbent, nas_data.incumbents["1 - Dice"]])
        if include_hnas and dataset in self._hnas_datasets:
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
            palette=sns.color_palette()[:min(3, n_baseline_approaches)],
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
            palette=sns.color_palette()[3: n_hpo_approaches + 3],
            ax=g
        )

        # We add markers to highlight the final values
        hpo_approaches = incumbent["Approach"].unique()
        for i in range(n_hpo_approaches):
            approach = hpo_approaches[i]
            grouped_approach = incumbent[incumbent["Approach"] == approach].groupby("Full Model Trainings")
            last_value = grouped_approach["1 - Dice"].mean().iloc[-1]

            color = sns.color_palette()[3 + i]
            g.scatter(
                self.n_full_trainings,
                last_value,
                color=color,
                zorder=5,
                linewidth=1,
                s=100,
                marker="*",
                edgecolor="black"
            )


        g.set_title(f"Optimization Process for {format_dataset_name(dataset)}")
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

        plt.minorticks_on()
        plt.grid(visible=True, which="both", linestyle="--")
        plt.grid(visible=True, which="minor", linestyle=":", linewidth=0.5)

        g.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=(n_baseline_approaches + n_hpo_approaches) // 2,
            fancybox=False,
            shadow=False,
            frameon=False
        )

        plt.tight_layout()
        plt.savefig(
            self.hpo_plots / f"{dataset}_{self.configuration}_{'hpo_nas' if include_nas else 'hpo'}.png",
            dpi=400
        )

        plt.clf()


    def plot_hpo(self, **kwargs) -> None:
        for dataset in self._hpo_datasets:
            try:
                self._plot_hpo_(dataset, **kwargs)
            except ValueError:
                self.logger.info(f"Unable to plot HPO for {dataset}.")
                continue

    def plot_hpo_combined(
            self,
            x_log_scale: bool = False,
            y_log_scale: bool = False,
            show_error: bool = False,
            include_nas: bool = False,
            include_hnas: bool = False
        ) -> None:

        fig, axes = plt.subplots(
            nrows=2,
            ncols=5,
            sharex=True,
            # sharey=True,
            figsize=(10, 5)
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
                palette=sns.color_palette()[:min(3, n_baseline_approaches)],
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
                palette=sns.color_palette()[3: n_hpo_approaches + 3],
                ax=ax
            )

            # We add markers to highlight the final values
            hpo_approaches = incumbent["Approach"].unique()
            for i in range(n_hpo_approaches):
                approach = hpo_approaches[i]
                grouped_approach = incumbent[incumbent["Approach"] == approach].groupby("Full Model Trainings")
                last_value = grouped_approach["1 - Dice"].mean().iloc[-1]
                last_x = grouped_approach["Full Model Trainings"].mean().iloc[-1]

                color = sns.color_palette()[3 + i]
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
        handles, labels = baseline_handles[:max_baseline_approaches] + hpo_handles[-max_hpo_approaches:], baseline_labels[:max_baseline_approaches] + hpo_labels[-max_hpo_approaches:]

        fig.subplots_adjust(
            top=0.91,
            bottom=0.15,
            left=0.07,
            right=0.99,
            hspace=0.3,
            wspace=0.35
        )

        axes[-3].legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.26),
            ncol=len(handles),
            fancybox=False,
            shadow=False,
            frameon=False
        )

        plt.savefig(
            self.hpo_plots / "hpo_combined.png" if not include_nas else self.hpo_plots / "hpo_nas_combined.png",
            dpi=400
        )

        plt.clf()

    def _plot_nas_(
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

        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.5))

        n_baseline_approaches = 0

        for baseline_approach in baseline_data.metrics_foreground_mean["Approach"].unique():
            metrics = baseline_data.metrics_foreground_mean
            metrics = metrics[metrics["Approach"] == baseline_approach]
            baseline_dice = metrics["Dice"].mean()

            baseline_progress = baseline_data.progress
            baseline_progress = baseline_progress[baseline_progress["Approach"] == baseline_approach]
            baseline_time = baseline_progress.groupby("Fold")["Runtime"].sum().mean() / 3600

            color = sns.color_palette()[n_baseline_approaches]

            sns.scatterplot(
                x=[1 - baseline_dice],
                y=[baseline_time],
                color=color,
                label=baseline_approach,
                ax=ax
            )

            n_baseline_approaches += 1

        hpo_data = self.get_hpo_data(dataset)

        # get mean of last 5 entries in incumbent
        hpo_dice = hpo_data.incumbent["1 - Dice"].iloc[-5:].mean()
        hpo_time = hpo_data.incumbent_progress.groupby("Fold")["Runtime"].sum().mean() / 3600

        color = sns.color_palette()[n_baseline_approaches]

        sns.scatterplot(
            x=[hpo_dice],
            y=[hpo_time],
            color=color,
            label="HPO (ours)",
            marker="x",
            ax=ax
        )

        history = nas_data.history
        pareto_front = history.sort_values(by="1 - Dice")
        pareto_front = pareto_front[pareto_front["Runtime"] == pareto_front["Runtime"].cummin()]

        # Round budget column
        history.loc[:, "Budget"] = history["Budget"].round()

        if show_configs:
            g = sns.scatterplot(
                data=history,
                x="1 - Dice",
                y="Runtime",
                # label="Configurations",
                color=sns.color_palette()[n_baseline_approaches + 1],
                size="Budget",
                s=10,
                alpha=0.5,
                ax=ax
            )

        g = sns.lineplot(
            data=pareto_front,
            x="1 - Dice",
            y="Runtime",
            color=sns.color_palette()[n_baseline_approaches + (2 if show_configs else 1)],
            label="HPO + NAS (ours)",
            ax=ax,
            drawstyle="steps-post"
        )

        g.set_title(format_dataset_name(dataset))
        g.set_xlabel("1 - Dice")
        g.set_ylabel("Training Runtime [h]")

        plt.grid(visible=True, which="both", linestyle="--")
        plt.grid(visible=True, which="minor", linestyle=":", linewidth=0.5)

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
        plt.tight_layout()
        if approach_key == "hpo_nas":
            dir = self.nas_plots
        else:
            dir = self.hnas_plots
        plt.savefig(
            dir / f"{dataset}_{self.configuration}{f'_nas' if show_configs else f'_nas_no_configs'}.png",
            dpi=400
        )

        plt.clf()

    def plot_nas_combined(
            self,
            approach_key: str,
            x_log_scale: bool = False,
            y_log_scale: bool = False
        ) -> None:

        fig, axes = plt.subplots(
            nrows=2,
            ncols=5,
            sharex=True,
            # sharey=True,
            figsize=(12, 6)
        )
        axes = axes.flatten()

        max_approaches = 0
        max_approaches_ax = axes[0]

        for ax, dataset in zip(axes, self._hpo_datasets, strict=False):
            if approach_key == "hpo_nas":
                nas_data = self.get_nas_data(dataset)
            elif approach_key == "hpo_hnas":
                nas_data = self.get_hnas_data(dataset)
            else:
                raise ValueError(f"Unknown approach key {approach_key}.")
            baseline_data = self.get_baseline_data(dataset)

            n_baseline_approaches = 0

            for baseline_approach in baseline_data.metrics_foreground_mean["Approach"].unique():
                metrics = baseline_data.metrics_foreground_mean
                metrics = metrics[metrics["Approach"] == baseline_approach]
                baseline_dice = metrics["Dice"].mean()
                baseline_time = baseline_data.progress["Runtime"][1:].mean()

                color = sns.color_palette()[n_baseline_approaches]

                # Plot lines for baseline objectives
                ax.axhline(y=baseline_time, color=color, linestyle="--", label=baseline_approach)
                ax.axvline(x=1 - baseline_dice, color=color, linestyle="--")

                n_baseline_approaches += 1

            history = nas_data.history
            if len(history) == 0:
                continue

            pareto_front = history.sort_values(by="1 - Dice")
            pareto_front = pareto_front[pareto_front["Runtime"] == pareto_front["Runtime"].cummin()]

            sns.lineplot(
                data=pareto_front,
                x="1 - Dice",
                y="Runtime",
                color=sns.color_palette()[2],
                label="Pareto Front",
                ax=ax,
                drawstyle="steps-post"
            )

            ax.set_title(format_dataset_name(dataset).replace(" ", "\n"))
            ax.set_xlabel("1 - Dice")
            ax.set_ylabel("Training Runtime [h]")

            ax.set_xscale("log")
            ax.set_yscale("log")

            if ax != axes[0] and ax != axes[5]:
                ax.set_ylabel("")

            ax.get_legend().remove()

        # We use the axis with most approaches to get the legend
        handles, labels = max_approaches_ax.get_legend_handles_labels()

        fig.subplots_adjust(
            top=0.92,
            bottom=0.15,
            left=0.06,
            right=0.99,
            hspace=0.3,
            wspace=0.35
        )

        axes[-3].legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.23),
            ncol=max_approaches + 1,
            fancybox=False,
            shadow=False,
            frameon=False
        )

        if approach_key == "hpo_nas":
            dir = self.nas_plots
        else:
            dir = self.hnas_plots
        plt.savefig(
            dir / f"nas_combined.png",
            dpi=400
        )

        plt.clf()

    def plot_nas(self, approach_key: str, **kwargs) -> None:
        for dataset in self._nas_datasets:
            try:
                self._plot_nas_(dataset, approach_key=approach_key, **kwargs)
            except ValueError as e:
                self.logger.error(e)
                self.logger.info(f"Unable to plot NAS for {dataset}.")
                continue

    def plot_hyperparameter_importances(
            self,
            budget: int | Literal["combined"] = "combined"
        ):
        fig, axes = plt.subplots(
            nrows=2,
            ncols=5,
            sharex=True,
            sharey=True,
            figsize=(12, 6)
        )
        axes = axes.flatten()

        for ax, dataset in zip(axes, self._hpo_datasets, strict=False):
            deepcave_run = self._hpo_data.deepcave_runs[dataset]

            budgets = deepcave_run.get_budgets()
            if budget == "combined":
                selected_budget = -1
            else:
                if budget not in range(len(budgets) - 1):
                    raise ValueError(f"Budget index {budget} not found in {budgets}.")
                selected_budget = float(budgets[budget])

            evaluator = fANOVA(run=deepcave_run)
            evaluator.calculate(n_trees=16, budget=selected_budget)
            importances = evaluator.get_importances(hp_names=list(deepcave_run.configspace.keys()))

            importances_data = []
            for hp_key, hp_name in HYPERPARAMETER_REPLACEMENT_MAP.items():
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
                hue_order=HYPERPARAMETER_REPLACEMENT_MAP.values(),
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
                capsize=5,
            )

            g.set_title(format_dataset_name(dataset).replace(" ", "\n"))
            g.set_yticklabels([])

            ax.legend().set_visible(False)

        axes[-1].legend().set_visible(False)

        fig.subplots_adjust(
            top=0.92,
            bottom=0.15,
            left=0.04,
            right=0.98,
            hspace=0.3,
            wspace=0.2
        )

        # Create a single legend
        axes[-3].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.21),
            ncol=4,
            fancybox=False,
            shadow=False,
            frameon=False
        )

        plt.savefig(
            self.hpo_plots / "hyperparameter_importances.png",
            dpi=400
        )

        plt.clf()

    def compute_emissions(self):
        baseline_emissions = self._baseline_data.emissions[["run_id", "Approach", "Fold", "Dataset", "emissions"]]
        hpo_emissions = self._hpo_data.emissions[["run_id", "Approach", "Fold", "Dataset", "emissions"]]
        nas_emissions = self._nas_data.emissions[["run_id", "Approach", "Fold", "Dataset", "emissions"]]
        hnas_emissions = self._hnas_data.emissions[["run_id", "Approach", "Fold", "Dataset", "emissions"]]

        emissions = pd.concat([baseline_emissions, hpo_emissions, nas_emissions, hnas_emissions])
        emissions["Dataset"] = emissions["Dataset"].apply(format_dataset_name)
        emissions_per_dataset = emissions.groupby(["Dataset", "Approach"])["emissions"].sum().reset_index()

        plt.figure(1, figsize=(self.figwidth, 4.5))
        sns.barplot(
            x="Dataset",
            y="emissions",
            data=emissions_per_dataset,
            hue="Approach"
        )
        plt.tight_layout()
        plt.savefig(AUTONNUNET_PLOTS / f"{self.configuration}_emissions.png", dpi=400)

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
            plt.axhline(y=avg_runtime, color=sns.color_palette()[current_approach], linestyle="--")
            current_approach += 1

        plt.title("Runtime per Approach and Dataset")
        g.set_xlabel("Dataset")
        g.set_ylabel("Runtime [h]")

        plt.tight_layout()
        plt.savefig(self.baseline_plots / f"{self.configuration}_runtimes.png", dpi=500)

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
            axes[0].axhline(y=avg_score, color=sns.color_palette()[current_approach], linestyle="--")
            avg_runtime = dataset_runtimes[dataset_runtimes["Approach"] == approach]["Runtime"].mean()
            axes[1].axhline(y=avg_runtime, color=sns.color_palette()[current_approach], linestyle="--")
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
        plt.savefig(self.baseline_plots / f"{self.configuration}_performances_and_runtimes.png", dpi=500)


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
            plt.close()

            fig, ax = plt.subplots(1, 1, figsize=(4, 4))

            ax.imshow(img, cmap="gray")
            ax.imshow(label, cmap="viridis", alpha=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")

            plt.tight_layout()
            plt.savefig(output_dir / f"{dataset}_label.png", dpi=500)
            plt.close()

    def _plot_footprint(self, dataset: str, approach: str, objective: str):
        deepcave_run = self._hpo_data.deepcave_runs[dataset]

        fp = Footprint(
            run=deepcave_run
        )

        fp.calculate()

       