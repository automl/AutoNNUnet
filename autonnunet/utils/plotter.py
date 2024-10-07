from __future__ import annotations

import ast
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from autonnunet.datasets import ALL_DATASETS
from autonnunet.utils import format_dataset_name, load_json
from autonnunet.utils.paths import (AUTONNUNET_OUTPUT, AUTONNUNET_PLOTS,
                                    AUTONNUNET_TABLES)

PROGRESS_REPLACEMENT_MAP = {
    "mean_fg_dice": "Mean Foreground Dice",
    "ema_fg_dice": "EMA Foreground Dice",
    "train_losses": "Training Loss",
    "val_losses": "Validation Loss",
    "lrs": "Learning Rate",
}

STYLES_TYPE = Literal["white", "dark", "whitegrid", "darkgrid", "ticks"]

PROGRESS_FILENAME = "progress.csv"
HISTORY_FILENAME = "history.csv"
INCUMBENT_FILENAME = "incumbent.csv"
VALIDATION_METRICS_FILENAME = "summary.json"
EMISSIONS_FILENAME = "emissions.csv"
DATASET_JSON_FILENAME = "dataset.json"

TRAINING_BUDGET = 1000


@dataclass
class BaselineResult:
    progress: pd.DataFrame
    metrics: pd.DataFrame
    emissions: pd.DataFrame


@dataclass
class HPOResult:
    history: pd.DataFrame
    incumbent: pd.DataFrame
    incumbent_progress: pd.DataFrame
    emissions: pd.DataFrame


class Plotter:
    def __init__(
            self,
            configuration: str,
            n_folds: int = 5,
            style: STYLES_TYPE = "whitegrid",
            palette: str = "colorblind",
            figsize: tuple = (8, 5),
            smac_seed: int = 0,
        ):
        # We need these to find the respective directories
        self.configuration = configuration

        self.n_folds = n_folds
        self.smac_seed = smac_seed

        # Directories
        self.baseline_dir =  AUTONNUNET_OUTPUT / "baseline"
        self.smac_mf_dir = AUTONNUNET_OUTPUT / "smac_mf"

        # Seaborn settings
        sns.set_style(style=style)
        sns.set_palette(palette=palette)

        # Matplotlib settings
        self.figsize = figsize
        # TODO add font size

        self.logger = logging.getLogger("Plotter")

        AUTONNUNET_PLOTS.mkdir(parents=True, exist_ok=True)
        AUTONNUNET_TABLES.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        self._baseline_data = self._load_baseline_data(datasets=ALL_DATASETS)
        self._hpo_data = self._load_hpo_data(datasets=ALL_DATASETS)

        self._baseline_datasets = self._baseline_data.progress["Dataset"].unique().tolist()
        self._hpo_datasets = self._hpo_data.incumbent_progress["Dataset"].unique().tolist()

        self.logger.info(f"Loaded {len(self._baseline_datasets)} datasets for baseline.")
        self.logger.info(f"Loaded {len(self._hpo_datasets)} datasets for HPO.")

    def get_baseline_data(self, dataset: str):
        progress = self._baseline_data.progress[self._baseline_data.progress["Dataset"] == dataset]
        emissions = self._baseline_data.emissions[self._baseline_data.emissions["Dataset"] == dataset]
        metrics = self._baseline_data.metrics[self._baseline_data.metrics["Dataset"] == dataset]

        return BaselineResult(
            progress=progress,
            emissions=emissions,
            metrics=metrics
        )

    def get_hpo_data(self, dataset: str):
        incumbent_progress = self._hpo_data.incumbent_progress[self._hpo_data.incumbent_progress["Dataset"] == dataset]
        emissions = self._hpo_data.emissions[self._hpo_data.emissions["Dataset"] == dataset]
        history = self._hpo_data.history[self._hpo_data.history["Dataset"] == dataset]
        incumbent = self._hpo_data.incumbent[self._hpo_data.incumbent["Dataset"] == dataset]

        return HPOResult(
            incumbent_progress=incumbent_progress,
            emissions=emissions,
            history=history,
            incumbent=incumbent
        )

    def _load_validation_metrics(self, fold_dir: Path) -> pd.DataFrame:
        metrics_path = fold_dir / "validation" / VALIDATION_METRICS_FILENAME
        dataset_info_path = fold_dir / DATASET_JSON_FILENAME

        metrics = load_json(metrics_path)
        dataset_info = load_json(dataset_info_path)
        labels = list(dataset_info["labels"].keys())

        rows = []

        for case in metrics["metric_per_case"]:
            prediction_file = Path(case["prediction_file"]).name
            reference_file = Path(case["reference_file"]).name

            for class_id, metrics in case["metrics"].items():
                row = {
                    "class_id": labels[int(class_id)],
                    **metrics,
                    "prediction_file": prediction_file,
                    "reference_file": reference_file
                }

                rows.append(row)

        return pd.DataFrame(rows)

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
        return pd.concat(
            [progress, labels_df],
            axis=1
        ).drop(columns=["dice_per_class_or_region"])


    def _load_baseline_data(self, datasets: list[str]):
        all_progress  = []
        all_emissions = []
        all_metrics = []

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
                metrics = self._load_validation_metrics(fold_dir)

                for df in [progress, emissions, metrics]:
                    df["Approach"] = "Baseline"
                    df["Fold"] = fold
                    df["Dataset"] = dataset

                all_progress.append(progress)
                all_emissions.append(emissions)
                all_metrics.append(metrics)

        all_progress = pd.concat(all_progress)
        all_emissions = pd.concat(all_emissions)
        all_metrics = pd.concat(all_metrics)

        return BaselineResult(
            progress=all_progress,
            metrics=all_metrics,
            emissions=all_emissions
        )

    def _load_incumbent(self, smac_run_dir: Path) -> pd.DataFrame:
        incumbent = pd.read_csv(smac_run_dir / INCUMBENT_FILENAME)
        history = pd.read_csv(smac_run_dir / HISTORY_FILENAME)

        incumbent_expanded = []
        for _, row in incumbent.iterrows():
            for fold, performance_key in enumerate([f"performance_fold_{i}" for i in range(5)]):
                performance = history[history["config_id"] == row["config_id"]][performance_key].values[0]

                row_data = {
                    "Configuration ID": row["config_id"],
                    "Cost": performance,
                    "Budget": row["budget"],
                    "Fold": fold,
                    "Budget Used": row["budget_used"]
                }

                incumbent_expanded.append(row_data)

        incumbent = pd.DataFrame(incumbent_expanded)
        incumbent["Full Model Trainings"] = incumbent["Budget Used"] / TRAINING_BUDGET

        return incumbent

    def _load_hpo_data(self, datasets: list[str]):
        all_progress  = []
        all_emissions = []
        all_history = []
        all_incumbent = []

        for dataset in datasets:
            dataset_dir = self.smac_mf_dir / dataset
            if not dataset_dir.exists():
                self.logger.info(f"SMAC: Skipping {dataset}.")
                continue

            smac_run_dir = dataset_dir \
                / self.configuration / str(self.smac_seed)
            if not (smac_run_dir / HISTORY_FILENAME).exists():
                self.logger.info(f"SMAC: Skipping {dataset}.")
                continue

            history = pd.read_csv(smac_run_dir / HISTORY_FILENAME)
            history["Dataset"] = dataset

            incumbent = self._load_incumbent(smac_run_dir)
            incumbent["Dataset"] = dataset

            all_history.append(history)
            all_incumbent.append(incumbent)

            incumbent_config_id = incumbent["Configuration ID"].values[-1]

            for config_id in history["config_id"].unique():
                for fold in range(self.n_folds):
                    run_id = config_id * self.n_folds + fold
                    run_dir = smac_run_dir / str(run_id)
                    if not (run_dir).exists():
                        continue

                    progress = self._load_progress(run_dir)
                    emissions = pd.read_csv(run_dir / EMISSIONS_FILENAME)

                    for df in [progress, emissions]:
                        df["Dataset"] = dataset
                        df["Approach"] = "Baseline"
                        df["Fold"] = fold
                        df["Configuration ID"] = config_id

                    all_emissions.append(emissions)
                    if config_id == incumbent_config_id:
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

    def plot_baseline(
            self,
            datasets: list[str],
            x_metric: str = "Epoch",
            y_metrics: list[str] | None = None,
        ):
        if y_metrics is None:
            y_metrics = ["mean_fg_dice", "val_losses"]
        training_results = self._load_baseline_data(datasets=datasets)

        fig, ax = plt.subplots(1, len(y_metrics), figsize=self.figsize)
        if len(y_metrics) == 1:
            ax = [ax]

        for i, y_metric in enumerate(y_metrics):
            sns.lineplot(
                x=x_metric,
                y=y_metric,
                data=training_results.progress,
                ax=ax[i]
            )

        plt.tight_layout()
        plt.savefig(AUTONNUNET_PLOTS / f"{self.configuration}_hpo.png", dpi=400)

    def _plot_hpo_(
            self,
            dataset: str
        ) -> None:
        hpo_data = self.get_hpo_data(dataset)
        baseline_data = self.get_baseline_data(dataset)

        plt.figure(1, figsize=self.figsize)

        # Group by Fold
        metrics = baseline_data.metrics.groupby(["Fold"])["Dice"].mean().reset_index()
        metrics_expanded = pd.DataFrame(
            np.repeat(metrics.values, 2, axis=0),
            columns=metrics.columns
        )
        metrics_expanded["Full Model Trainings"] = np.tile([0, 16], len(metrics))
        metrics_expanded["Cost"] = 1 - metrics_expanded["Dice"]

        g = sns.lineplot(
            data=metrics_expanded,
            x="Full Model Trainings",
            y="Cost",
            label="Baseline",
            linestyle="--",
            errorbar=("ci", 95)
        )

        sns.lineplot(
            x="Full Model Trainings",
            y="Cost",
            data=hpo_data.incumbent,
            drawstyle="steps-post",
            label="SMAC + HB",
            errorbar=("ci", 95)
        )

        g.set_title(f"Optimization Process for {format_dataset_name(dataset)}")
        g.set_xlabel("Full Model Trainings")
        g.set_ylabel("Cost")

        # log scale
        g.set_xlim(0, 16)
        # g.set_xscale("log")
        # g.set_yscale("log")

        plt.minorticks_on()
        plt.grid(visible=True, which="both", linestyle="--")
        plt.grid(visible=True, which="minor", linestyle=":", linewidth=0.5)

        plt.legend()

        plt.tight_layout()
        plt.savefig(AUTONNUNET_PLOTS / f"{dataset}_{self.configuration}_hpo.png", dpi=400)

        plt.clf()

    def plot_hpo(self) -> None:
        for dataset in self._hpo_datasets:
            self._plot_hpo_(dataset)

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
        baseline.metrics = baseline.metrics.groupby(["Dataset", "Fold", "Approach"]).mean().reset_index()
        hpo.metrics = hpo.metrics.groupby(["Dataset", "Fold", "Approach"]).mean().reset_index()

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

    def create_table(self, datasets: list[str]):
        baseline = self._load_baseline_data(datasets=datasets)
        hpo = self._load_baseline_data(datasets=datasets)
        hpo.metrics["Approach"] = "SMAC MF"

        print(baseline.metrics)
        sys.exit()

        # We remove all background classes, since we compute the mean foreground Dice
        baseline.metrics = baseline.metrics[baseline.metrics["class_id"] != 0]
        hpo.metrics = hpo.metrics[hpo.metrics["class_id"] != 0]

        # We can drop all other metrics
        baseline.metrics = baseline.metrics[["Dataset", "Fold", "Approach", "Dice"]]
        hpo.metrics = hpo.metrics[["Dataset", "Fold", "Approach", "Dice"]]

        # Then, we average over all prediction files
        baseline.metrics = baseline.metrics.groupby(["Dataset", "Fold", "Approach"]).mean().reset_index()
        hpo.metrics = hpo.metrics.groupby(["Dataset", "Fold", "Approach"]).mean().reset_index()

        metrics = pd.concat([baseline.metrics, hpo.metrics])
        metrics["Dataset"] = metrics["Dataset"].apply(format_dataset_name)

        grouped_metrics = metrics.groupby(["Dataset", "Approach"])["Dice"].agg(["mean", "std"]).reset_index()

        # First, we add the mean and standard deviation
        grouped_metrics["mean"] = grouped_metrics["mean"].round(2)
        grouped_metrics["std"] = grouped_metrics["std"].round(2)
        grouped_metrics["mean_std"] = grouped_metrics["mean"].astype(str) + " $\\pm$ " + grouped_metrics["std"].astype(str)

        # We highlight the best approach per dataset
        max_mean = grouped_metrics.groupby("Dataset")["mean"].transform("max") == grouped_metrics["mean"]
        grouped_metrics["mean_std"] = grouped_metrics.apply(lambda row: "\\textbf{" + row["mean_std"] + "}" if max_mean.loc[row.name] else row["mean_std"], axis=1)

        table = grouped_metrics.pivot(index="Dataset", columns="Approach", values="mean_std")
        table.to_latex(AUTONNUNET_TABLES / f"{self.configuration}_table.tex")



