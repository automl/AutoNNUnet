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
HISTORY_FILENAME = "runhistory.csv"
INCUMBENT_FILENAME = "incumbent.csv"
VALIDATION_METRICS_FILENAME = "summary.json"
EMISSIONS_FILENAME = "emissions.csv"
DATASET_JSON_FILENAME = "dataset.json"

TRAINING_BUDGET = 1000
N_FULL_TRAININGS = 13.816
BUDGETS = {
    0: [15.625, 62.5, 250, 1000],
    1: [62.5, 250, 1000],
    2: [250, 1000],
    3: [1000]
}

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
            figsize: tuple = (8, 3),
            hpo_seed: int = 0,
        ):
        # We need these to find the respective directories
        self.configuration = configuration

        self.n_folds = n_folds
        self.hpo_seed = hpo_seed

        # Directories
        self.baseline_dir =  AUTONNUNET_OUTPUT / "baseline"
        self.smac_dir = AUTONNUNET_OUTPUT / "smac_mf"
        self.prior_band_dir = AUTONNUNET_OUTPUT / "prior_band"

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
        self._smac_data = self._load_hpo_data(datasets=ALL_DATASETS, approach="smac")
        # self._prior_band_data = self._load_hpo_data(datasets=ALL_DATASETS, approach="prior_band")
        # self._prior_band_data = pd.DataFrame()

        self._baseline_datasets = self._baseline_data.progress["Dataset"].unique().tolist()
        self._smac_datasets = self._smac_data.incumbent_progress["Dataset"].unique().tolist()
        # self._prior_band_datasets = self._prior_band_data.incumbent_progress["Dataset"].unique().tolist()

        self.logger.info(
            f"Loaded {len(self._baseline_datasets)} datasets for baseline.")
        self.logger.info(
            f"Loaded {len(self._smac_datasets)} datasets for SMAC.")
        # self.logger.info(
        #     f"Loaded {len(self._prior_band_datasets)} datasets for PriorBand.")


    def get_baseline_data(self, dataset: str):
        progress = self._baseline_data.progress[
            self._baseline_data.progress["Dataset"] == dataset]
        emissions = self._baseline_data.emissions[
            self._baseline_data.emissions["Dataset"] == dataset]
        metrics = self._baseline_data.metrics[
            self._baseline_data.metrics["Dataset"] == dataset]

        return BaselineResult(
            progress=progress,
            emissions=emissions,
            metrics=metrics
        )

    def get_hpo_data(self, dataset: str):
        smac_incumbent_progress = self._smac_data.incumbent_progress[
            self._smac_data.incumbent_progress["Dataset"] == dataset]
        smac_emissions = self._smac_data.emissions[
            self._smac_data.emissions["Dataset"] == dataset]
        smac_history = self._smac_data.history[
            self._smac_data.history["Dataset"] == dataset]
        smac_incumbent = self._smac_data.incumbent[
            self._smac_data.incumbent["Dataset"] == dataset]

        # prior_band_incumbent_progress = self._prior_band_data.incumbent_progress[
        #     self._prior_band_data.incumbent_progress["Dataset"] == dataset]
        # prior_band_emissions = self._prior_band_data.emissions[
        #     self._prior_band_data.emissions["Dataset"] == dataset]
        # prior_band_history = self._prior_band_data.history[
        #     self._prior_band_data.history["Dataset"] == dataset]
        # prior_band_incumbent = self._prior_band_data.incumbent[
        #     self._prior_band_data.incumbent["Dataset"] == dataset]

        # incumbent_progress = pd.concat(
        #     [smac_incumbent_progress, prior_band_incumbent_progress])
        # emissions = pd.concat([smac_emissions, prior_band_emissions])
        # history = pd.concat([smac_history, prior_band_history])
        # incumbent = pd.concat([smac_incumbent, prior_band_incumbent])

        return HPOResult(
            incumbent_progress=smac_incumbent_progress,
            emissions=smac_emissions,
            history=smac_history,
            incumbent=smac_incumbent
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

        # Since we run succesive halving, we have to calculate the actual
        # budget of a run by subtracting the budget of the previous run
        incumbent.loc[:, "real_budget"] = 0.
        current_bracket_idx = 0
        current_stage_idx = 0
        prev_budget = BUDGETS[0][0]
        for row_idx, row in incumbent.iterrows():
            # We move to the next stage of the bracket
            if row["budget"] > prev_budget:
                current_stage_idx += 1

            # Now we transition to the next bracket
            elif row["budget"] < prev_budget:
                current_bracket_idx += 1
                current_stage_idx = 0

            # We are at the first stage of the bracket, nothing to do
            if current_stage_idx == 0:
                incumbent.loc[row_idx, "real_budget"] = row["budget"]
                prev_budget = row["budget"]
                continue

            # We are in the last bracket, nothing to do here
            if len(BUDGETS[current_bracket_idx]) == 0:
                incumbent.loc[row_idx, "real_budget"] = row["budget"]
                continue

            incumbent.loc[row_idx, "real_budget"] = round(BUDGETS[current_bracket_idx][current_stage_idx]) - round(BUDGETS[current_bracket_idx][current_stage_idx - 1])

            prev_budget = row["budget"]

        # The real used budget is the sum of all additional budgets
        incumbent["budget_used"] = incumbent["real_budget"].cumsum()

        incumbent_expanded = []
        for _, row in incumbent.iterrows():
            for fold, performance_key in enumerate(
                [f"performance_fold_{i}" for i in range(5)]
            ):
                performance = history[
                    history["config_id"] == row["config_id"]
                ][performance_key].to_numpy()[0]

                row_data = {
                    "Configuration ID": row["config_id"],
                    "Cost": performance,
                    "Budget": row["budget"],
                    "Fold": fold,
                    "Budget Used": row["budget_used"]
                }

                incumbent_expanded.append(row_data)

        incumbent = pd.DataFrame(incumbent_expanded)

        # Used budget is the cumulative sum of all additional budgets
        # Since one training consists of N_FOLDS, we divide by N_FOLDS
        incumbent["Full Model Trainings"] = incumbent["Budget Used"] / TRAINING_BUDGET

        return incumbent

    def _load_hpo_data(self, datasets: list[str], approach: str):
        assert approach in ["smac", "prior_band"]

        all_progress  = []
        all_emissions = []
        all_history = []
        all_incumbent = []

        base_dir = self.smac_dir if approach == "smac" else self.prior_band_dir
        approach = "SMAC + HB" if approach == "smac" else "PriorBand"

        for dataset in datasets:
            dataset_dir = base_dir / dataset
            if not dataset_dir.exists():
                self.logger.info(f"{approach}: Skipping {dataset}.")
                continue

            smac_run_dir = dataset_dir \
                / self.configuration / str(self.hpo_seed)
            if not (smac_run_dir / HISTORY_FILENAME).exists():
                self.logger.info(f"{approach}: Skipping {dataset}.")
                continue

            history = pd.read_csv(smac_run_dir / HISTORY_FILENAME)
            history["Dataset"] = dataset
            history["Approach"] = approach

            incumbent = self._load_incumbent(smac_run_dir)
            incumbent["Dataset"] = dataset
            incumbent["Approach"] = approach

            all_history.append(history)
            all_incumbent.append(incumbent)

            incumbent_config_id = incumbent["Configuration ID"].to_numpy()[-1]

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
                        df["Approach"] = approach
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
            dataset: str
        ) -> None:
        hpo_data = self.get_hpo_data(dataset)
        baseline_data = self.get_baseline_data(dataset)

        plt.figure(1, figsize=self.figsize)

        metrics = baseline_data.metrics.groupby(["Fold"])["Dice"].mean().reset_index()
        metrics_expanded = pd.DataFrame(
            np.repeat(metrics.values, 2, axis=0),
            columns=metrics.columns
        )
        metrics_expanded["Full Model Trainings"] = np.tile(
            [0, N_FULL_TRAININGS],
            len(metrics)
        )
        metrics_expanded["Cost"] = 1 - metrics_expanded["Dice"]

        n_approaches = len(hpo_data.incumbent["Approach"].unique())

        g = sns.lineplot(
            data=metrics_expanded,
            x="Full Model Trainings",
            y="Cost",
            label="Baseline",
            linestyle="--",
            errorbar=("sd")
        )

        sns.lineplot(
            x="Full Model Trainings",
            y="Cost",
            data=hpo_data.incumbent,
            drawstyle="steps-post",
            hue="Approach",
            errorbar=("sd"),
            palette=sns.color_palette()[1: n_approaches + 1]
        )

        g.set_title(f"Optimization Process for {format_dataset_name(dataset)}")
        g.set_xlabel("Full Model Trainings")
        g.set_ylabel("Cost")

        # log scale
        g.set_xlim(0, N_FULL_TRAININGS)
        g.set_xticks(np.arange(0, N_FULL_TRAININGS + 1, 5))
        # g.set_xscale("log")
        # g.set_yscale("log")

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
            AUTONNUNET_PLOTS / f"{dataset}_{self.configuration}_hpo.png",
            dpi=400
        )

        plt.clf()

    def plot_hpo(self) -> None:
        for dataset in self._smac_datasets:
            try:
                self._plot_hpo_(dataset)
            except ValueError:
                continue

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



