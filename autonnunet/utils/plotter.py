from __future__ import annotations

import ast
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from autonnunet.utils import (compute_hyperband_budgets, format_dataset_name,
                              get_budget_per_config, load_json)
from autonnunet.utils.paths import (AUTONNUNET_OUTPUT, AUTONNUNET_PLOTS,
                                    AUTONNUNET_TABLES)

PROGRESS_REPLACEMENT_MAP = {
    "mean_fg_dice": "Mean Foreground Dice",
    "ema_fg_dice": "EMA Foreground Dice",
    "train_losses": "Training Loss",
    "val_losses": "Validation Loss",
    "lrs": "Learning Rate",
}

HISTORY_REPLACEMENT_MAP = {
    "config_id": "Configuration ID",
    "run_id": "Run ID",
    "budget": "Budget",
    "o0_loss": "1 - Dice",
    "o1_epoch_runtime": "Epoch Runtime",
}

STYLES_TYPE = Literal["white", "dark", "whitegrid", "darkgrid", "ticks"]

OBJECTIVES_MAPPING = {
    "1 - Dice": "loss",
    "Epoch Runtime": "epoch_runtime"
}

PROGRESS_FILENAME = "progress.csv"
HISTORY_FILENAME = "runhistory.csv"
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

@dataclass
class NASResult:
    history: pd.DataFrame
    incumbents: dict[str, pd.DataFrame]
    emissions: pd.DataFrame

class Plotter:
    def __init__(
            self,
            configuration: str,
            datasets: list[str],
            objectives: list[str] = ["1 - Dice", "Epoch Runtime"],
            min_budget: float = 10.0,
            max_budget: float = 1000.0,
            eta: int = 3,
            n_folds: int = 5,
            style: STYLES_TYPE = "whitegrid",
            palette: str = "colorblind",
            figsize: tuple = (8, 3),
            hpo_seed: int = 0,
        ):
        self.datasets = datasets
        self.objectives = objectives

        # We need these to find the respective directories
        self.configuration = configuration

        self.min_budget = min_budget
        self.max_budget = max_budget

        self.n_folds = n_folds
        self.hpo_seed = hpo_seed

        # Hyperband configuration
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
            is_prior_band=True  
        )
        self.n_full_trainings = self.total_real_budget / self.max_budget
        # self.n_full_trainings = 5

        # We need this to assign the real budget for each configuration
        self.real_budgets_per_config = get_budget_per_config(
            n_configs_in_stage=self.n_configs_in_stage,
            budgets_in_stage=self.real_budgets_in_stage
        )

        # Directories
        self.baseline_dir =  AUTONNUNET_OUTPUT / "baseline"
        self.hpo_dir = AUTONNUNET_OUTPUT / "prior_band"
        self.nas_dir = AUTONNUNET_OUTPUT / "prior_band_nas"

        # Seaborn settings
        sns.set_style(style=style)
        sns.set_palette(palette=palette)
        self.palette = palette

        # Matplotlib settings
        self.figsize = figsize
        # TODO add font size

        self.logger = logging.getLogger("Plotter")

        self.hpo_plots = AUTONNUNET_PLOTS / "hpo"
        self.hpo_plots.mkdir(parents=True, exist_ok=True)
        self.nas_plots = AUTONNUNET_PLOTS / "nas"
        self.nas_plots.mkdir(parents=True, exist_ok=True)
        self.baseline_plots = AUTONNUNET_PLOTS / "baseline"
        self.baseline_plots.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        self._baseline_data = self._load_baseline_data(datasets=self.datasets)
        # self._hpo_data = self._load_hpo_data(datasets=self.datasets)
        # self._nas_data = self._load_nas_data(datasets=self.datasets)

        self._baseline_datasets = self._baseline_data.progress["Dataset"].unique().tolist()
        # self._hpo_datasets = self._hpo_data.history["Dataset"].unique().tolist()
        # self._nas_datasets = self._nas_data.history["Dataset"].unique().tolist()

        self.logger.info(
            f"Loaded {len(self._baseline_datasets)} datasets for baseline.")
        # self.logger.info(
        #     f"Loaded {len(self._hpo_datasets)} datasets for HPO.")
        # self.logger.info(
        #     f"Loaded {len(self._nas_datasets)} datasets for NAS.")

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
            incumbent=prior_band_incumbent
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
            incumbents=incumbents
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

        for dataset in datasets:
            dataset_dir = self.baseline_dir / dataset
            if not dataset_dir.exists():
                self.logger.info(f"Skipping {dataset}.")
                continue

            for fold in range(self.n_folds):
                fold_dir = dataset_dir \
                      / self.configuration / f"fold_{fold}"
                if not (fold_dir).exists():
                    self.logger.info(f"Skipping {fold_dir} of dataset {dataset}.")
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
                    df["Approach"] = "Baseline"
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
            run_dir: Path,
            filename: str = INCUMBENT_FILENAME,
            objective: str = "o0_loss"
        ) -> pd.DataFrame:
        incumbent = pd.read_csv(run_dir / filename)
        history = pd.read_csv(run_dir / HISTORY_FILENAME)

        # Since we run succesive halving, we have to insert the real
        # budget of a run by subtracting the budget of the previous run
        # in the runhistory
        history.loc[:, "real_budget"] = 0.

        for run_id, real_budget in self.real_budgets_per_config.items():
            history.loc[history["run_id"] == run_id, "real_budget"] = real_budget

        # The real used budget is the sum of all additional budgets
        incumbent["real_budget_used"] = history["real_budget"].cumsum()

        assert len(incumbent) == len(history)

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
                    "1 - Dice": performance,
                    "Budget": row["budget"],
                    "Fold": fold,
                    "Budget Used": row["budget_used"],
                    "Real Budget Used": row["real_budget_used"],
                }

                incumbent_expanded.append(row_data)

        incumbent = pd.DataFrame(incumbent_expanded)

        # Used budget is the cumulative sum of all additional budgets
        # Since one training consists of N_FOLDS, we divide by N_FOLDS
        incumbent["Full Model Trainings"] = incumbent["Real Budget Used"] / self.max_budget

        return incumbent

    def _load_history(self, history_path: Path) -> pd.DataFrame:
        history = pd.read_csv(history_path)
        return history.rename(columns=HISTORY_REPLACEMENT_MAP)

    def _load_nas_data(self, datasets: list[str]) -> NASResult:
        all_emissions = []
        all_history = []
        all_incumbent = defaultdict(list)

        approach = "PriorBand NAS"

        for dataset in datasets:
            dataset_dir = self.nas_dir / dataset
            if not dataset_dir.exists():
                self.logger.info(f"{approach}: Skipping {dataset}.")
                continue

            nas_run_dir = dataset_dir \
                / self.configuration / str(self.hpo_seed)
            if not (nas_run_dir / HISTORY_FILENAME).exists():
                self.logger.info(f"{approach}: Skipping {dataset}.")
                continue

            history = self._load_history(nas_run_dir / HISTORY_FILENAME)
            history["Dataset"] = dataset
            history["Approach"] = approach

            for i, objective in enumerate(self.objectives):
                incumbent_filename = f"incumbent_{OBJECTIVES_MAPPING[objective]}.csv"
                obj_name = f"o{i}_{OBJECTIVES_MAPPING[objective]}"

                incumbent = self._load_incumbent(
                    run_dir=nas_run_dir,
                    filename=incumbent_filename,
                    objective=obj_name
                )
                incumbent["Dataset"] = dataset
                incumbent["Approach"] = approach

                all_incumbent[objective].append(incumbent)

            all_history.append(history)

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
            emissions=all_emissions
        )

    def _load_hpo_data(self, datasets: list[str]) -> HPOResult:
        all_progress  = []
        all_emissions = []
        all_history = []
        all_incumbent = []

        approach = "PriorBand"

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

            history = self._load_history(hpo_run_dir / HISTORY_FILENAME)
            history["Dataset"] = dataset
            history["Approach"] = approach

            incumbent = self._load_incumbent(hpo_run_dir)
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
            emissions=all_emissions
        )

    def _plot_baseline(self, dataset: str) -> None:
        baseline_data = self.get_baseline_data(dataset)

        plt.figure(1, figsize=self.figsize)

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
            AUTONNUNET_PLOTS / f"{dataset}_{self.configuration}_baseline.png",
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
            y_log_scale: bool = False
        ) -> None:
        hpo_data = self.get_hpo_data(dataset)
        baseline_data = self.get_baseline_data(dataset)

        plt.figure(1, figsize=self.figsize)

        metrics = baseline_data.metrics_foreground_mean

        metrics_expanded = pd.DataFrame(
            np.repeat(metrics.values, 2, axis=0),
            columns=metrics.columns
        )
        metrics_expanded["Full Model Trainings"] = np.tile(
            [0, self.n_full_trainings],
            len(metrics)
        )
        metrics_expanded["1 - Dice"] = 1 - metrics_expanded["Dice"]

        n_approaches = len(hpo_data.incumbent["Approach"].unique())

        g = sns.lineplot(
            data=metrics_expanded,
            x="Full Model Trainings",
            y="1 - Dice",
            label="Baseline",
            linestyle="--",
            errorbar=("sd")
        )

        sns.lineplot(
            x="Full Model Trainings",
            y="1 - Dice",
            data=hpo_data.incumbent,
            drawstyle="steps-post",
            hue="Approach",
            errorbar=("sd"),
            palette=sns.color_palette()[1: n_approaches + 1]
        )

        g.set_title(f"Optimization Process for {format_dataset_name(dataset)}")
        g.set_xlabel("Full Model Trainings")
        g.set_ylabel("1 - Dice")

        # g.set_xticks(np.arange(0, N_FULL_TRAININGS + 1, 5))

        if x_log_scale:
            g.set_xscale("log")
            g.set_xlim(0.1, self.n_full_trainings)
        else:
            g.set_xlim(0, self.n_full_trainings)

        if y_log_scale:
            g.set_ylim(1e-2, 1)
            g.set_yscale("log")
        else:
            pass
            # g.set_ylim(0.9 * baseline_mean, None)

        plt.minorticks_on()
        plt.grid(visible=True, which="both", linestyle="--")
        plt.grid(visible=True, which="minor", linestyle=":", linewidth=0.5)

        g.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=n_approaches + 1,
            fancybox=False,
            shadow=False,
            frameon=False
        )

        plt.tight_layout(rect=(0, -0.07, 1, 1))
        plt.savefig(
            self.hpo_plots / f"{dataset}_{self.configuration}_hpo.png",
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

    def _plot_nas_(
            self,
            dataset: str,
        ) -> None:
        nas_data = self.get_nas_data(dataset)
        baseline_data = self.get_baseline_data(dataset)

        plt.figure(1, figsize=self.figsize)

        metrics = baseline_data.metrics_foreground_mean
        baseline_dice = metrics["Dice"].mean()
        baseline_time = baseline_data.progress["Runtime"][1:].mean()

        history = nas_data.history

        g = sns.scatterplot(
            x=[1 - baseline_dice],
            y=[baseline_time],
            color="red",
            s=100,
            label="Baseline",
        )

        g = sns.scatterplot(
            data=history,
            x="1 - Dice",
            y="Epoch Runtime",
            size="Budget",
            ax=g
        )

        pareto_front = history.sort_values(by="1 - Dice")
        pareto_front = pareto_front[pareto_front["Epoch Runtime"] == pareto_front["Epoch Runtime"].cummin()]

        sns.lineplot(
            data=pareto_front,
            x="1 - Dice",
            y="Epoch Runtime",
            ax=g,
        )

        g.set_title(f"Optimization Process for {format_dataset_name(dataset)}")
        g.set_xlabel("1 - Dice")
        g.set_ylabel("Epoch Runtime [s]")

        g.set_xscale("log")
        g.set_yscale("log")

        plt.legend()
        plt.tight_layout()
        plt.savefig(
            self.nas_plots / f"{dataset}_{self.configuration}_nas.png",
            dpi=400
        )

        plt.clf()

    def plot_nas(self, **kwargs) -> None:
        for dataset in self._nas_datasets:
            self._plot_nas_(dataset, **kwargs)

            try:
                self._plot_nas_(dataset, **kwargs)
            except ValueError as e:
                self.logger.info(f"Unable to plot NAS for {dataset}.")
                continue

    def _plot_hyperband(self, dataset: str) -> None:
        hpo_data = self.get_hpo_data(dataset)

        plt.figure(1, figsize=self.figsize)

        history = hpo_data.history.copy()
        history["Budget"] = history["Budget"].apply(lambda x: round(x))

        budgets = sorted(np.unique([round(b) for b in self.budgets_in_stage[0]]))
        for budget_key, budget in enumerate(budgets):
            history.loc[history["Budget"] == budget, "Budget"] = budget_key + 1

        g = sns.lineplot(
            x="Run ID",
            y="Budget",
            data=history,
            drawstyle="steps-post",
        )

        g.set_xlabel("Number of Trials")

        all_configs = [v for sublist in self.n_configs_in_stage.values() for v in sublist]
        n_configs = sum(all_configs)

        x_ticks = [next(iter(self.n_configs_in_stage.values()))[0], *np.cumsum([sum(sublist) for sublist in list(self.n_configs_in_stage.values())[:-1]]).tolist()]

        g.set_xlim(0, n_configs + 1)
        g.set_xticks(x_ticks)

        g.set_ylim(0.9, len(self.budgets_in_stage[0]) + 0.1)
        g.set_yticks(range(1, len(budgets) + 1))
        g.set_yticklabels([str(b) for b in budgets])

        plt.tight_layout()
        plt.savefig(AUTONNUNET_PLOTS / f"{dataset}_{self.configuration}_hyperband.png", dpi=400)

        plt.clf()

    def plot_hyperband(self) -> None:
        for dataset in self._hpo_datasets:
            self._plot_hyperband(dataset)

    def plot_metrics(
            self,
            datasets: list[str],
            metric: Literal["Dice", "IoU"] = "Dice"
            ) -> None:
        baseline = self._load_baseline_data(datasets=datasets)
        hpo = self._load_baseline_data(datasets=datasets)
        hpo.metrics["Approach"] = "SMAC MF"

        # We can drop all other metrics
        baseline.metrics = baseline.metrics[["Dataset", "Fold", "Approach", metric]]
        hpo.metrics = hpo.metrics[["Dataset", "Fold", "Approach", metric]]

        # Then, we average over all prediction files
        baseline.metrics = baseline.metrics.groupby(
            ["Dataset", "Fold", "Approach"]).mean().reset_index()
        hpo.metrics = hpo.metrics.groupby(
            ["Dataset", "Fold", "Approach"]).mean().reset_index()

        metrics = pd.concat([baseline.metrics, hpo.metrics])

        # Now, we can create the boxplot the metrics with variance over folds
        plt.figure(1, figsize=self.figsize)
        sns.boxplot(
            x="Dataset",
            y=metric,
            data=metrics,
            hue="Approach",
        )
        plt.tight_layout()
        plt.savefig(AUTONNUNET_PLOTS / f"{self.configuration}_metrics.png", dpi=400)

    def compute_emissions(self):
        baseline_emissions = self._baseline_data.emissions[["run_id", "Approach", "Fold", "Dataset", "emissions"]]
        hpo_emissions = self._smac_data.emissions[["run_id", "Approach", "Fold", "Dataset", "emissions"]]

        emissions = pd.concat([baseline_emissions, hpo_emissions])
        emissions["Dataset"] = emissions["Dataset"].apply(format_dataset_name)
        emissions_per_dataset = emissions.groupby(["Dataset", "Approach"])["emissions"].sum().reset_index()

        plt.figure(1, figsize=self.figsize)
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
        hpo = self._load_baseline_data(datasets=datasets)
        hpo.metrics["Approach"] = "SMAC MF"

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
        baseline_runtimes = self._baseline_data.progress[["Dataset", "Fold", "Runtime"]]
        dataset_runtimes = baseline_runtimes.groupby(["Dataset", "Fold"]).sum().reset_index()
        dataset_runtimes["Runtime"] /= 3600

        dataset_runtimes["Dataset"] = dataset_runtimes["Dataset"].apply(lambda s: format_dataset_name(s).replace(" ", "\n"))

        # Create bar plot of dataset runtimes and mark average
        # Use different colors for each bar
        plt.figure(1, figsize=(self.figsize[0] * 1.5, self.figsize[1] * 1.3))
        g = sns.barplot(
            x="Dataset",
            y="Runtime",
            data=dataset_runtimes,
        )

        avg_runtime = dataset_runtimes["Runtime"].mean()
        plt.axhline(y=avg_runtime, color=sns.color_palette(self.palette)[1], linestyle="--")

        plt.title(f"Runtime per Dataset (Mean = {avg_runtime:.2f} h)")
        g.set_xlabel("Dataset")
        g.set_ylabel("Runtime [h]")

        plt.tight_layout()
        plt.savefig(AUTONNUNET_PLOTS / f"{self.configuration}_runtimes.png", dpi=500)

        print(dataset_runtimes.groupby("Dataset")["Runtime"].mean().reset_index()["Runtime"] * 60)


