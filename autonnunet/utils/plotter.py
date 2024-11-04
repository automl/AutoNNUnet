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

from autonnunet.utils import compute_hyperband_budgets, get_real_budget_per_config
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

HISTORY_REPLACEMENT_MAP = {
    "config_id": "Configuration ID",
    "run_id": "Run ID",
    "budget": "Budget",
}

STYLES_TYPE = Literal["white", "dark", "whitegrid", "darkgrid", "ticks"]

PROGRESS_FILENAME = "progress.csv"
HISTORY_FILENAME = "runhistory.csv"
INCUMBENT_FILENAME = "incumbent.csv"
VALIDATION_METRICS_FILENAME = "summary.json"
EMISSIONS_FILENAME = "emissions.csv"
DATASET_JSON_FILENAME = "dataset.json"

# print full dataframes
pd.set_option("display.max_rows", None)

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
            datasets: list[str],
            min_budget: float = 4.0,
            max_budget: float = 1000.0,
            n_full_trainings: int = 12,
            eta: int = 3,
            n_folds: int = 5,
            style: STYLES_TYPE = "whitegrid",
            palette: str = "colorblind",
            figsize: tuple = (8, 3),
            hpo_seed: int = 0,
        ):
        self.datasets = datasets

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
            _
        ) = compute_hyperband_budgets(
            b_min=min_budget,
            b_max=max_budget,
            eta=eta,
            print_output=False,
            is_prior_band=True  # DEBUG
        )

        self.n_full_trainings = n_full_trainings

        self.real_budgets_per_config = get_real_budget_per_config(
            n_configs_in_stage=self.n_configs_in_stage,
            real_budgets_in_stage=self.real_budgets_in_stage
        )

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
        self._baseline_data = self._load_baseline_data(datasets=self.datasets)
        # self._smac_data = self._load_hpo_data(datasets=self.datasets, approach="smac")
        # self._prior_band_data = self._load_hpo_data(datasets=self.datasets, approach="prior_band")

        self._baseline_datasets = self._baseline_data.progress["Dataset"].unique().tolist()
        # self._smac_datasets = self._smac_data.incumbent_progress["Dataset"].unique().tolist()
        # self._prior_band_datasets = self._prior_band_data.incumbent_progress["Dataset"].unique().tolist()

        self.logger.info(
            f"Loaded {len(self._baseline_datasets)} datasets for baseline.")
        # self.logger.info(
        #     f"Loaded {len(self._smac_datasets)} datasets for SMAC.")
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
        # smac_incumbent_progress = self._smac_data.incumbent_progress[
        #     self._smac_data.incumbent_progress["Dataset"] == dataset]
        # smac_emissions = self._smac_data.emissions[
        #     self._smac_data.emissions["Dataset"] == dataset]
        # smac_history = self._smac_data.history[
        #     self._smac_data.history["Dataset"] == dataset]
        # smac_incumbent = self._smac_data.incumbent[
        #     self._smac_data.incumbent["Dataset"] == dataset]

        prior_band_incumbent_progress = self._prior_band_data.incumbent_progress[
            self._prior_band_data.incumbent_progress["Dataset"] == dataset]
        prior_band_emissions = self._prior_band_data.emissions[
            self._prior_band_data.emissions["Dataset"] == dataset]
        prior_band_history = self._prior_band_data.history[
            self._prior_band_data.history["Dataset"] == dataset]
        prior_band_incumbent = self._prior_band_data.incumbent[
            self._prior_band_data.incumbent["Dataset"] == dataset]
        
        return HPOResult(
            incumbent_progress=prior_band_incumbent_progress,
            emissions=prior_band_emissions,
            history=prior_band_history,
            incumbent=prior_band_incumbent
        )

        # incumbent_progress = pd.concat(
        #     [smac_incumbent_progress, prior_band_incumbent_progress])
        # emissions = pd.concat([smac_emissions, prior_band_emissions])
        # history = pd.concat([smac_history, prior_band_history])
        # incumbent = pd.concat([smac_incumbent, prior_band_incumbent])

        # return HPOResult(
        #     incumbent_progress=incumbent_progress,
        #     emissions=emissions,
        #     history=history,
        #     incumbent=incumbent
        # )

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

    def _load_incumbent(self, hpo_run_dir: Path) -> pd.DataFrame:
        incumbent = pd.read_csv(hpo_run_dir / INCUMBENT_FILENAME)
        history = pd.read_csv(hpo_run_dir / HISTORY_FILENAME)

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
                [f"performance_fold_{i}" for i in range(5)]
            ):
                performance = history[
                    history["run_id"] == row["run_id"]
                ][performance_key].to_numpy()[0]

                row_data = {
                    "Run ID": row["run_id"],
                    "Cost": performance,
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
        history = history.rename(columns=HISTORY_REPLACEMENT_MAP)
        return history

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

        metrics = baseline_data.metrics.groupby(["Fold"])["Dice"].mean().reset_index()

        metrics_expanded = pd.DataFrame(
            np.repeat(metrics.values, 2, axis=0),
            columns=metrics.columns
        )
        metrics_expanded["Full Model Trainings"] = np.tile(
            [0, self.n_full_trainings],
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

        # g.set_xticks(np.arange(0, N_FULL_TRAININGS + 1, 5))

        if x_log_scale:
            g.set_xscale("log")
        g.set_xlim(1, self.n_full_trainings)

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
            AUTONNUNET_PLOTS / f"{dataset}_{self.configuration}_hpo.png",
            dpi=400
        )

        plt.clf()

    def plot_hpo(self, **kwargs) -> None:
        for dataset in self._prior_band_datasets:
            try:
                self._plot_hpo_(dataset, **kwargs)
            except ValueError as e:
                self.logger.info(f"Skipping {dataset}.")
                continue

    def _plot_hyperband(self, dataset: str) -> None:
        hpo_data = self.get_hpo_data(dataset)
        
        plt.figure(1, figsize=self.figsize)

        history = hpo_data.history.copy()

        for budget_key, budget in enumerate(self.budgets_in_stage[0]):
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

        x_ticks = [
            0,
            list(self.n_configs_in_stage.values())[0][0]
        ] + np.cumsum([sum(sublist) for sublist in list(self.n_configs_in_stage.values())[:-1]]).tolist()

        g.set_xlim(1, n_configs + 1)
        g.set_xticks(x_ticks)
        
        g.set_ylim(0.9, len(self.budgets_in_stage[0]) + 0.1)
        g.set_yticks(range(1, len(self.budgets_in_stage[0]) + 1))
        g.set_yticklabels([str(b) for b in self.budgets_in_stage[0]])

        plt.tight_layout()
        plt.savefig(AUTONNUNET_PLOTS / f"{dataset}_{self.configuration}_hyperband.png", dpi=400)

        plt.clf()

    def plot_hyperband(self) -> None:
        for dataset in self._prior_band_datasets:
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



