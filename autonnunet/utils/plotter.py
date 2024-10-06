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


@dataclass
class TrainingResult:
    progress: pd.DataFrame
    metrics: pd.DataFrame
    emissions: pd.DataFrame


@dataclass
class HPOResult:
    history: pd.DataFrame
    incumbent: pd.DataFrame
    progress: pd.DataFrame
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

    def _load_validation_metrics(self, path: Path) -> pd.DataFrame:
        metrics_path = path / "validation" / VALIDATION_METRICS_FILENAME
        dataset_info_path = path / DATASET_JSON_FILENAME

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
        progress = pd.concat(
            [progress, labels_df],
            axis=1
        ).drop(columns=["dice_per_class_or_region"])

        return progress

    def _load_baseline_data(self, datasets: list[str]):
        all_progress  = []
        all_emissions = []
        all_metrics = []

        for dataset in datasets:
            for fold in range(self.n_folds):
                fold_dir = self.baseline_dir / dataset \
                      / self.configuration / f"fold_{fold}"
                if not (fold_dir).exists():
                    self.logger.info(f"Skipping {fold_dir}.")
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

        return TrainingResult(
            progress=all_progress,
            metrics=all_metrics,
            emissions=all_emissions
        )

    def _load_hpo_data(self, datasets: list[str]):
        all_progress  = []
        all_emissions = []
        all_history = []
        all_incumbent = []

        for dataset in datasets:
            smac_run_dir = self.smac_mf_dir / dataset \
                / self.configuration / str(self.smac_seed)
            history = pd.read_csv(smac_run_dir / HISTORY_FILENAME)
            incumbent = pd.read_csv(smac_run_dir / INCUMBENT_FILENAME)
            history["Dataset"] = dataset
            incumbent["Dataset"] = dataset

            incumbent["Full Model Trainings"] = incumbent["budget_used"] / 1000

            all_history.append(history)
            all_incumbent.append(incumbent)

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
                        df["Configuration"] = config_id

                    all_progress.append(progress)
                    all_emissions.append(emissions)

        all_progress = pd.concat(all_progress)
        all_emissions = pd.concat(all_emissions)
        all_history = pd.concat(all_history)
        all_incumbent = pd.concat(all_incumbent)

        return HPOResult(
            history=all_history,
            incumbent=all_incumbent,
            progress=all_progress,
            emissions=all_emissions
        )

    def plot_baseline(
            self,
            datasets: list[str],
            x_metric: str = "epoch",
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
            hpo_data: HPOResult, 
            baseline_data: TrainingResult, 
            dataset: str
        ) -> None:
        incumbent_data = hpo_data.incumbent[hpo_data.incumbent["Dataset"] == dataset]

        plt.figure(1, figsize=self.figsize)

        g = sns.lineplot(
            x="Full Model Trainings",
            y="performance",
            data=incumbent_data,
            drawstyle="steps-post",
        )

        baseline_data_dataset = baseline_data.metrics[baseline_data.metrics["Dataset"] == dataset]
        baseline_performance = baseline_data_dataset["Dice"].mean()
        baseline_cost = 1 - baseline_performance

        # add horizontal line showing mean validation performance
        g.axhline(baseline_cost, color="red", linestyle="--")
        
        g.set_title("Optimization Process")
        g.set_xlabel("Full Model Trainings")
        g.set_ylabel("Cost")

        # log scale
        g.set_xscale("log")
        # g.set_yscale("log")

        plt.minorticks_on()
        plt.grid(visible=True, which="both", linestyle="--")
        plt.grid(visible=True, which="minor", linestyle=":", linewidth=0.5)

        plt.tight_layout()
        plt.savefig(AUTONNUNET_PLOTS / f"{dataset}_{self.configuration}_hpo.png", dpi=400)

        plt.clf()

    def plot_hpo(self, datasets: list[str]) -> None:
        baseline_data = self._load_baseline_data(datasets=datasets)
        hpo_data = self._load_hpo_data(datasets=datasets)

        for dataset in datasets:
            self._plot_hpo_(hpo_data, baseline_data, dataset)

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



