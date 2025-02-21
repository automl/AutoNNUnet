"""The plotter for the AutoNNUNet analysis."""
from __future__ import annotations

import contextlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib
import matplotlib.axis
import matplotlib.colors as mcolors
import matplotlib.figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from ConfigSpace import CategoricalHyperparameter
from deepcave.constants import COMBINED_BUDGET
from deepcave.evaluators.ablation import Ablation
from deepcave.evaluators.footprint import Footprint
from deepcave.evaluators.lpi import LPI
from deepcave.evaluators.mo_ablation import MOAblation
from deepcave.evaluators.mo_fanova import MOfANOVA
from deepcave.evaluators.mo_lpi import MOLPI
from deepcave.plugins.budget.budget_correlation import BudgetCorrelation
from deepcave.plugins.hyperparameter.pdp import PartialDependencies
from deepcave.utils.styled_plotty import get_hyperparameter_ticks
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator
from tqdm import tqdm

from autonnunet.analysis.fanova import fANOVA
from autonnunet.analysis.dataset_features import extract_dataset_features
from autonnunet.analysis.deepcave_utils import runhistory_to_deepcave
from autonnunet.datasets import ALL_DATASETS
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
    "baseline_medsam2": "MedSAM2",
    "hpo": "HPO (ours)",
    "hpo_nas": "HPO+NAS (ours)",
    "hpo_hnas": "HPO+HNAS (ours)"
}

NNUNET_PROGRESS_REPLACEMENT_MAP = {
    "mean_fg_dice": "Mean Foreground Dice [%]",
    "ema_fg_dice": "EMA Foreground Dice [%]",
    "train_losses": "Training Loss",
    "val_losses": "Validation Loss",
    "lrs": "Learning Rate",
}

O_DSC = "1 - DSC [%]"
O_RUNTIME = "Runtime [h]"

OBJECTIVES_MAPPING = {
    O_DSC: "loss",
    O_RUNTIME: "runtime"
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

HYPERPARAMETER_VALUE_REPLACE_MAP = {
    "conv_encoder": "Convolutional",
    "ConvolutionalEncoder": "Convolutional",
    "res_encoder": "Residual",
    "ResidualEncoderM": "Residual",
    "dropout": "Dropout",
    "no_dropout": "No Dropout",
    "instance_norm": "Instance Norm.",
    "batch_norm": "Batch Norm.",
    "leaky_relu": "Leaky ReLU",
    "relu": "ReLU",
    "elu": "ELU",
    "gelu": "GELU",
    "prelu": "PReLU",
}

HISTORY_REPLACEMENT_MAP = {
    "config_id": "Configuration ID",
    "run_id": "Run ID",
    "budget": "Budget",
    "o0_loss": O_DSC,
    "o1_runtime": O_RUNTIME,
}

DATASET_FEATURES_REPLACEMENT_MAP = {
    "Dataset": "Dataset",     # This is only here for the filtering based on keys()
    "instance": "Instance",
    "class_idx": "Class Index",
    "class_label": "Class Label",
    "volume": "Volume",
    "class_volume": "Class Volume",
    "class_volume_ratio": "Class Volume Ratio",
    "mean_intensity": "Intensity Mean",
    "std_intensity": "Intensity Std.",
    "min_intensity": "Intensity Min.",
    "max_intensity": "Intensity Max.",
    "n_images": "#Images",
    "n_classes": "#Classes",
    "source": "Source",
}

STYLES_TYPE = Literal["white", "dark", "whitegrid", "darkgrid", "ticks"]

ORIGIN_MAP = {
    "random": "Rand. Sampling",
    "prior": "Prior Sampling",
    "incumbent": "Inc. Sampling",
}

PROGRESS_FILENAME = "progress.csv"
HISTORY_FILENAME = "runhistory.csv"
SAMPLING_POLICY_LOGS = "sampling_policy.log"
INCUMBENT_FILENAME = "incumbent_loss.csv"
NNUNET_VALIDATION_METRICS_FILENAME = "summary.json"
MEDSAM2_VALIDATION_METRICS_FILENAME = "validation_results.csv"
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

@dataclass
class BaselineResult:
    """A dataclass to store the baseline results."""
    progress: pd.DataFrame
    metrics: pd.DataFrame
    emissions: pd.DataFrame

@dataclass
class HPOResult:
    """A dataclass to store the HPO results."""
    history: pd.DataFrame
    incumbent: pd.DataFrame
    incumbent_progress: pd.DataFrame
    incumbent_metrics: pd.DataFrame
    emissions: pd.DataFrame
    deepcave_runs: dict[str, DeepCAVERun]

@dataclass
class NASResult:
    """A dataclass to store the NAS results."""
    history: pd.DataFrame
    incumbents: dict[str, pd.DataFrame]
    incumbent_metrics: pd.DataFrame
    incumbent_progress: pd.DataFrame
    emissions: pd.DataFrame
    deepcave_runs: dict[str, DeepCAVERun]

class Plotter:
    """A comprehensive plotter for the AutoNNUNET analysis.

    Parameters
    ----------
    datasets : list[str]
        The datasets to analyze.

    configuration : str
        The nnU-Net configuration name. Defaults to "3d_fullres".

    objectives : list[str] | None
        The objectives to analyze. Defaults to None.

    min_budget : float
        The minimum budget for the Hyperband configuration. Defaults to 10.0.

    max_budget : float
        The maximum budget for the Hyperband configuration. Defaults to 1000.0.

    eta : int
        The eta value for the Hyperband configuration. Defaults to 3.

    n_folds : int
        The number of folds. Defaults to 5.

    style : STYLES_TYPE
        The Seaborn style. Defaults to "whitegrid".

    color_palette : list
        The color palette. Defaults to sns.color_palette("colorblind").

    figwidth : int
        The figure width. Defaults to 8.

    hpo_seed : int
        The HPO seed. Defaults to 0.

    dpi : int
        The DPI value. Defaults to 400.

    format : str
        The format of the plots. Defaults to "png".

    lazy_loading : bool
        Whether to lazy load the data. Defaults to True.
    """
    def __init__(       # noqa: PLR0913
            self,
            datasets: list[str] = ALL_DATASETS,
            configuration: str = "3d_fullres",
            objectives: list[str] | None = None,
            min_budget: float = 10.0,
            max_budget: float = 1000.0,
            eta: int = 3,
            n_folds: int = 5,
            style: STYLES_TYPE = "whitegrid",
            color_palette: list | None = None,
            figwidth: int = 8,
            hpo_seed: int = 0,
            dpi: int = 400,
            file_format: str = "png",
            lazy_loading: bool = True       # noqa: FBT001, FBT002
        ):
        """Initializes the plotter.

        Parameters
        ----------
        datasets : list[str]
            The datasets to analyze.

        configuration : str
            The nnU-Net configuration name. Defaults to "3d_fullres".

        objectives : list[str] | None
            The objectives to analyze. Defaults to None. In that case,
            the objectives are set to [O_DSC, O_RUNTIME].

        min_budget : float
            The minimum budget for the Hyperband configuration. Defaults to 10.0.

        max_budget : float
            The maximum budget for the Hyperband configuration. Defaults to 1000.0.

        eta : int
            The eta value for the Hyperband configuration. Defaults to 3.

        n_folds : int
            The number of folds. Defaults to 5.

        style : STYLES_TYPE
            The Seaborn style. Defaults to "whitegrid".

        color_palette : list
            The color palette. Defaults to sns.color_palette("colorblind").

        figwidth : int
            The figure width. Defaults to 8.

        hpo_seed : int
            The HPO seed. Defaults to 0.

        dpi : int
            The DPI value. Defaults to 400.

        file_format : str
            The format of the plots. Defaults to "png".

        lazy_loading : bool
            Whether to lazy load the data. Defaults to True.
        """
        self.logger = logging.getLogger("Plotter")

        if objectives is None:
            objectives = [O_DSC, O_RUNTIME]
        self.datasets = datasets
        self.objectives = objectives

        # We need these to find the corresponding directories
        self.configuration = configuration
        self.n_folds = n_folds
        self.hpo_seed = hpo_seed

        self._setup_hyperband(min_budget=min_budget, max_budget=max_budget, eta=eta)
        self._setup_paths()

        # Seaborn settings
        sns.set_style(
            style=style,
            rc={
                "axes.edgecolor": "black",
                "axes.linewidth": 1.0,
                "xtick.color": "black",
                "ytick.color": "black",
                "xtick.bottom": True,
                "ytick.left": True
            }
        )
        self.figwidth = figwidth
        self.dpi = dpi
        self.format = file_format

        if color_palette is None:
            self.color_palette = sns.color_palette("colorblind")
        else:
            self.color_palette = color_palette

        self._init_data()
        if not lazy_loading:
            self.load_all_data()

    def _init_data(self):
        """Initialites all data structures."""
        self._baseline_data = BaselineResult(
            progress=pd.DataFrame(),
            metrics=pd.DataFrame(),
            emissions=pd.DataFrame()
        )
        self._baseline_datasets = []

        self._hpo_data = HPOResult(
            history=pd.DataFrame(),
            incumbent=pd.DataFrame(),
            incumbent_progress=pd.DataFrame(),
            incumbent_metrics=pd.DataFrame(),
            emissions=pd.DataFrame(),
            deepcave_runs={}
        )
        self._hpo_datasets = []

        self._nas_data = NASResult(
            history=pd.DataFrame(),
            incumbents={k: pd.DataFrame() for k in self.objectives},
            incumbent_metrics=pd.DataFrame(),
            incumbent_progress=pd.DataFrame(),
            emissions=pd.DataFrame(),
            deepcave_runs={}
        )
        self._nas_datasets = []

        self._hnas_data = NASResult(
            history=pd.DataFrame(),
            incumbents={k: pd.DataFrame() for k in self.objectives},
            incumbent_metrics=pd.DataFrame(),
            incumbent_progress=pd.DataFrame(),
            emissions=pd.DataFrame(),
            deepcave_runs={}
        )
        self._hnas_datasets = []

    def _setup_hyperband(self, min_budget: float, max_budget: float, eta: int) -> None:
        """Computes the Hyperband configuration.

        Parameters
        ----------
        min_budget : float
            The minimum budget.

        max_budget : float
            The maximum budget.

        eta : int
            The reduction factor.
        """
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

    def _setup_paths(self) -> None:
        """Creates the necessary directories for the plots."""
        self.baseline_conv =  AUTONNUNET_OUTPUT / "baseline_ConvolutionalEncoder"
        self.baseline_resenc_m =  AUTONNUNET_OUTPUT / "baseline_ResidualEncoderM"
        self.baseline_resenc_l =  AUTONNUNET_OUTPUT / "baseline_ResidualEncoderL"
        self.baseline_medsam2 =  AUTONNUNET_OUTPUT / "baseline_medsam2"
        self.hpo_dir = AUTONNUNET_OUTPUT / "hpo"
        self.nas_dir = AUTONNUNET_OUTPUT / "hpo_nas"
        self.hnas_dir = AUTONNUNET_OUTPUT / "hpo_hnas"
        self.cross_eval_dir = AUTONNUNET_OUTPUT / "cross_eval"

        self.combined_plots = AUTONNUNET_PLOTS / "combined"
        self.combined_plots.mkdir(parents=True, exist_ok=True)
        self.nas_plots = AUTONNUNET_PLOTS / "hpo_nas"
        self.nas_plots.mkdir(parents=True, exist_ok=True)
        self.hnas_plots = AUTONNUNET_PLOTS / "hpo_hnas"
        self.hnas_plots.mkdir(parents=True, exist_ok=True)
        self.baseline_plots = AUTONNUNET_PLOTS / "baseline"
        self.baseline_plots.mkdir(parents=True, exist_ok=True)

        self.dataset_analysis_plots = AUTONNUNET_PLOTS / "analysis" / "datasets"
        self.dataset_analysis_plots.mkdir(parents=True, exist_ok=True)
        self.hpo_analysis_plots = AUTONNUNET_PLOTS / "analysis" / "hpo"
        self.hpo_analysis_plots.mkdir(parents=True, exist_ok=True)
        self.nas_analysis_plots = AUTONNUNET_PLOTS / "analysis" / "hpo_nas"
        self.nas_analysis_plots.mkdir(parents=True, exist_ok=True)
        self.hnas_analysis_plots = AUTONNUNET_PLOTS / "analysis" / "hpo_hnas"
        self.hnas_analysis_plots.mkdir(parents=True, exist_ok=True)

        self.analysis_plots = {
            "hpo": self.hpo_analysis_plots,
            "hpo_nas": self.nas_analysis_plots,
            "hpo_hnas": self.hnas_analysis_plots
        }

        AUTONNUNET_TABLES.mkdir(exist_ok=True)

    def _load_baseline_data_lazy(self, dataset: str) -> None:
        """Lazy loading of the baseline data.

        Parameters
        ----------
        dataset : str
            The dataset name.
        """
        if dataset in self._baseline_datasets:
            return

        baseline_data = self._load_baseline_data(datasets=[dataset])
        self._baseline_data.progress = pd.concat(
            [self._baseline_data.progress, baseline_data.progress])
        self._baseline_data.emissions = pd.concat(
            [self._baseline_data.emissions, baseline_data.emissions])
        self._baseline_data.metrics = pd.concat(
            [self._baseline_data.metrics, baseline_data.metrics])

    def _load_hpo_data_lazy(self, dataset: str) -> None:
        """Lazy loading of the HPO data.

        Parameters
        ----------
        dataset : str
            The dataset name.
        """
        if dataset in self._hpo_datasets:
            return

        hpo_data = self._load_hpo_data(datasets=[dataset])
        self._hpo_data.history = pd.concat(
            [self._hpo_data.history, hpo_data.history])
        self._hpo_data.incumbent = pd.concat(
            [self._hpo_data.incumbent, hpo_data.incumbent])
        self._hpo_data.incumbent_progress = pd.concat(
            [self._hpo_data.incumbent_progress, hpo_data.incumbent_progress])
        self._hpo_data.incumbent_metrics = pd.concat(
            [self._hpo_data.incumbent_metrics, hpo_data.incumbent_metrics])
        self._hpo_data.emissions = pd.concat(
            [self._hpo_data.emissions, hpo_data.emissions])
        self._hpo_data.deepcave_runs.update(hpo_data.deepcave_runs)
        self._hpo_datasets = [] if len(self._hpo_data.history) == 0 \
            else self._hpo_data.history["Dataset"].unique().tolist()

    def _load_nas_data_lazy(self, dataset: str, approach_key: str) -> None:
        """Lazy loading of the NAS data.

        Parameters
        ----------
        dataset : str
            The dataset name.

        approach_key : str
            The approach key (hpo, hpo_nas, hpo_hnas).
        """
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
        data.incumbents = {k: pd.concat(
            [v, nas_data.incumbents[k]]) for k, v in data.incumbents.items()}
        data.incumbent_metrics = pd.concat(
            [data.incumbent_metrics, nas_data.incumbent_metrics])
        data.incumbent_progress = pd.concat(
            [data.incumbent_progress, nas_data.incumbent_progress])
        data.emissions = pd.concat([data.emissions, nas_data.emissions])
        data.deepcave_runs.update(nas_data.deepcave_runs)

        if approach_key == "hpo_nas":
            self._nas_datasets = [] \
                if len(data.history) == 0 else data.history["Dataset"].unique().tolist()
        else:
            self._hnas_datasets = [] \
                if len(data.history) == 0 else data.history["Dataset"].unique().tolist()

    def load_all_data(self) -> None:
        """Loads data for all datasets at once."""
        self._baseline_data = self._load_baseline_data(datasets=self.datasets)
        self._baseline_datasets = self._baseline_data.progress[
            "Dataset"].unique().tolist()
        self.logger.info(
            f"Loaded {len(self._baseline_datasets)} datasets for baseline.")

        self._hpo_data = self._load_hpo_data(datasets=self.datasets)
        self._hpo_datasets = self._hpo_data.history["Dataset"].unique().tolist()
        self.logger.info(
            f"Loaded {len(self._hpo_datasets)} datasets for HPO.")

        self._nas_data = self._load_nas_data(
            datasets=self.datasets,
            approach_key="hpo_nas"
        )
        self._nas_datasets = self._nas_data.history["Dataset"].unique().tolist()
        self.logger.info(
            f"Loaded {len(self._nas_datasets)} datasets for HPO + NAS.")

        self._hnas_data = self._load_nas_data(
            datasets=self.datasets,
            approach_key="hpo_hnas"
        )
        self._hnas_datasets = self._nas_data.history["Dataset"].unique().tolist()
        self.logger.info(
            f"Loaded {len(self._hnas_datasets)} datasets for HPO + HNAS.")

    def get_baseline_data(self, dataset: str) -> BaselineResult:
        """Returns the baseline data for a specific dataset.

        Parameters
        ----------
        dataset : str
            The dataset name.

        Returns:
        -------
        BaselineResults
            The baseline results.
        """
        if dataset not in self._baseline_datasets:
            self._load_baseline_data_lazy(dataset=dataset)

        progress = self._baseline_data.progress[
            self._baseline_data.progress["Dataset"] == dataset]
        emissions = self._baseline_data.emissions[
            self._baseline_data.emissions["Dataset"] == dataset]
        metrics = self._baseline_data.metrics[
            self._baseline_data.metrics["Dataset"] == dataset]

        return BaselineResult(
            progress=progress,
            emissions=emissions,
            metrics=metrics,
        )

    def get_hpo_data(self, dataset: str) -> HPOResult:
        """Returns the HPO data for a specific dataset.

        Parameters
        ----------
        dataset : str
            The dataset name.

        Returns:
        -------
        HPOResult
            The HPO results.
        """
        if dataset not in self._hpo_datasets:
            self._load_hpo_data_lazy(dataset=dataset)

        if dataset not in self._hpo_datasets:
            # We have no data for this dataset
            return HPOResult(
                history=pd.DataFrame(),
                incumbent=pd.DataFrame(),
                incumbent_progress=pd.DataFrame(),
                incumbent_metrics=pd.DataFrame(),
                emissions=pd.DataFrame(),
                deepcave_runs={}
            )

        incumbent_progress = self._hpo_data.incumbent_progress[
            self._hpo_data.incumbent_progress["Dataset"] == dataset]
        incumbent_metrics = self._hpo_data.incumbent_metrics[
            self._hpo_data.incumbent_metrics["Dataset"] == dataset]
        emissions = self._hpo_data.emissions[
            self._hpo_data.emissions["Dataset"] == dataset]
        history = self._hpo_data.history[
            self._hpo_data.history["Dataset"] == dataset]
        incumbent = self._hpo_data.incumbent[
            self._hpo_data.incumbent["Dataset"] == dataset]

        return HPOResult(
            incumbent_progress=incumbent_progress,
            incumbent_metrics=incumbent_metrics,
            emissions=emissions,
            history=history,
            incumbent=incumbent,
            deepcave_runs={dataset: self._hpo_data.deepcave_runs[dataset]}
        )

    def get_nas_data(self, dataset: str) -> NASResult:
        """Returns the NAS data for a specific dataset.

        Parameters
        ----------
        dataset : str
            The dataset name.

        Returns:
        -------
        NASResult
            The NAS results.
        """
        if dataset not in self._nas_datasets:
            self._load_nas_data_lazy(dataset=dataset, approach_key="hpo_nas")

        if dataset not in self._nas_datasets:
            # We have no data for this dataset
            return NASResult(
                emissions=pd.DataFrame(),
                history=pd.DataFrame(),
                incumbents={k: pd.DataFrame() for k in self.objectives},
                incumbent_metrics=pd.DataFrame(),
                incumbent_progress=pd.DataFrame(),
                deepcave_runs={}
            )

        emissions = self._nas_data.emissions[
            self._nas_data.emissions["Dataset"] == dataset]
        history = self._nas_data.history[
            self._nas_data.history["Dataset"] == dataset]
        incumbents = {}
        for objective in self.objectives:
            incumbents[objective] = self._nas_data.incumbents[objective][
                self._nas_data.incumbents[objective]["Dataset"] == dataset]
        incumbent_metrics = self._nas_data.incumbent_metrics[
            self._nas_data.incumbent_metrics["Dataset"] == dataset]
        incumbent_progress = self._nas_data.incumbent_progress[
            self._nas_data.incumbent_progress["Dataset"] == dataset]

        return NASResult(
            emissions=emissions,
            history=history,
            incumbents=incumbents,
            incumbent_metrics=incumbent_metrics,
            incumbent_progress=incumbent_progress,
            deepcave_runs={dataset: self._nas_data.deepcave_runs[dataset]}
        )

    def get_hnas_data(self, dataset: str) -> NASResult:
        """Returns the HNAS data for a specific dataset.

        Parameters
        ----------
        dataset : str
            The dataset name.

        Returns:
        -------
        NASResult
            The HNAS results.
        """
        if dataset not in self._hnas_datasets:
            self._load_nas_data_lazy(dataset=dataset, approach_key="hpo_hnas")

        if dataset not in self._hnas_datasets:
            # We have no data for this dataset
            return NASResult(
                emissions=pd.DataFrame(),
                history=pd.DataFrame(),
                incumbents={k: pd.DataFrame() for k in self.objectives},
                incumbent_metrics=pd.DataFrame(),
                incumbent_progress=pd.DataFrame(),
                deepcave_runs={}
            )

        emissions = self._hnas_data.emissions[
            self._hnas_data.emissions["Dataset"] == dataset]
        history = self._hnas_data.history[
            self._hnas_data.history["Dataset"] == dataset]
        incumbents = {}
        for objective in self.objectives:
            incumbents[objective] = self._hnas_data.incumbents[objective][
                self._hnas_data.incumbents[objective]["Dataset"] == dataset]
        deepcave_runs = {dataset: self._hnas_data.deepcave_runs[dataset]}
        incumbent_metrics = self._hnas_data.incumbent_metrics[
            self._hnas_data.incumbent_metrics["Dataset"] == dataset]
        incumbent_progress = self._hnas_data.incumbent_progress[
            self._hnas_data.incumbent_progress["Dataset"] == dataset]

        return NASResult(
            emissions=emissions,
            history=history,
            incumbents=incumbents,
            incumbent_metrics=incumbent_metrics,
            incumbent_progress=incumbent_progress,
            deepcave_runs=deepcave_runs
        )

    def get_deepcave_data(
            self,
            dataset: str,
            approach_key: str,
            objective: str = O_DSC
        ) -> tuple[DeepCAVERun, pd.DataFrame, pd.DataFrame]:
        """Returns the DeepCave data for a specific dataset.

        Parameters
        ----------
        dataset : str
            The dataset name.

        approach_key : str
            The approach key (hpo, hpo_nas, hpo_hnas).

        objective : str
            The objective to analyze. Defaults to "1 - DSC [%]".

        Returns:
        -------
        tuple[DeepCAVERun, pd.DataFrame, pd.DataFrame]
            The DeepCave run, the history, and the incumbent.
        """
        if approach_key == "hpo":
            deepcave_run = self.get_hpo_data(dataset).deepcave_runs[dataset]
            history = self.get_hpo_data(dataset).history
            incumbent = self.get_hpo_data(dataset).incumbent
        elif approach_key == "hpo_nas":
            deepcave_run = self.get_nas_data(dataset).deepcave_runs[dataset]
            history = self.get_nas_data(dataset).history
            incumbent = self.get_nas_data(dataset).incumbents[objective]
        elif approach_key == "hpo_hnas":
            deepcave_run = self.get_hnas_data(dataset).deepcave_runs[dataset]
            history = self.get_hnas_data(dataset).history
            incumbent = self.get_hnas_data(dataset).incumbents[objective]
        else:
            raise ValueError(f"Unknown approach key {approach_key}.")

        return deepcave_run, history, incumbent

    def _load_nnunet_metrics(self, fold_dir: Path) -> pd.DataFrame:
        """Load the validation metrics for nnU-Net.

        Parameters
        ----------
        fold_dir : Path
            The fold directory where the model was trained.

        Returns:
        -------
        pd.DataFrame
            The validation metrics.
        """
        metrics_path = fold_dir / "validation" / NNUNET_VALIDATION_METRICS_FILENAME
        dataset_info_path = fold_dir / DATASET_JSON_FILENAME

        validation_metrics = load_json(metrics_path)
        dataset_info = load_json(dataset_info_path)
        labels = {str(v): k for k, v in dataset_info["labels"].items()}

        metrics_df = pd.DataFrame({
            labels[k]: [v["Dice"] * 100] for k, v in validation_metrics["mean"].items()
        })
        metrics_df["Mean"] = [validation_metrics["foreground_mean"]["Dice"] * 100]

        return metrics_df

    def _load_medsam2_metrics(self, fold_dir: Path) -> pd.DataFrame:
        """Load the validation metrics for MedSAM2.

        Parameters
        ----------
        fold_dir : Path
            The fold directory where the model was trained.

        Returns:
        -------
        pd.DataFrame
            The validation metrics.
        """
        metrics_path = fold_dir / MEDSAM2_VALIDATION_METRICS_FILENAME
        if not metrics_path.exists():
            return pd.DataFrame()

        validation_metrics = pd.read_csv(metrics_path)
        dataset_info_path = self.baseline_conv / fold_dir.parts[-2] /\
            self.configuration / fold_dir.parts[-1] / DATASET_JSON_FILENAME

        dataset_info = load_json(dataset_info_path)
        labels = {str(v): k for k, v in dataset_info["labels"].items()}

        validation_metrics = validation_metrics.drop(
            columns=["case"]
        ).mean().to_frame().T
        validation_metrics = validation_metrics.rename(columns=labels)
        validation_metrics["Mean"] = validation_metrics.mean(axis=1)

        for label in labels.values():
            if label not in validation_metrics.columns:
                continue
            validation_metrics.loc[:, label] = validation_metrics[label] * 100
        validation_metrics.loc[:, "Mean"] = validation_metrics.loc[:, "Mean"] * 100

        return validation_metrics

    def _load_nnunet_progress(self, fold_dir: Path) -> pd.DataFrame:
        """Load the training progress for nnU-Net.

        Parameters
        ----------
        fold_dir : Path
            The fold directory where the model was trained.

        Returns:
        -------
        pd.DataFrame
            The progress.
        """
        dataset_info = load_json(fold_dir / DATASET_JSON_FILENAME)

        labels = list(dataset_info["labels"].keys())
        if labels[0] == "background":
            labels = labels[1:]

        path = fold_dir / PROGRESS_FILENAME

        progress = pd.read_csv(path)
        progress["Epoch"] = np.arange(len(progress))

        progress["Runtime"] = progress["epoch_end_timestamps"] \
            - progress["epoch_start_timestamps"]
        progress = progress[[
            "Epoch",
            "mean_fg_dice",
            "ema_fg_dice",
            "train_losses",
            "val_losses",
            "Runtime"
        ]]

        progress = progress.rename(columns=NNUNET_PROGRESS_REPLACEMENT_MAP)

        progress["Mean Foreground Dice [%]"] = progress[
            "Mean Foreground Dice [%]"] * 100
        progress["EMA Foreground Dice [%]"] = progress[
            "EMA Foreground Dice [%]"] * 100

        return progress


    def _load_medsam2_progress(self, fold_dir: Path) -> pd.DataFrame:
        """Load the training progress for MedSAM2.

        Parameters
        ----------
        fold_dir : Path
            The fold directory where the model was trained.

        Returns:
        -------
        pd.DataFrame
            The progress.
        """
        path = fold_dir / PROGRESS_FILENAME
        if not path.exists():
            return pd.DataFrame()

        progress = pd.read_csv(path)

        progress = progress.rename(columns={
            "Epoch Runtime": "Runtime"
        })

        progress["Mean Foreground Dice [%]"] = (
            1 - progress["Validation Segmentation Loss"]
        ) * 100
        progress["EMA Foreground Dice [%]"] = (
            1 - progress["Validation Segmentation Loss"]
        ) * 100

        for i in range(1, len(progress)):
            progress.loc[i, "EMA Foreground Dice [%]"] = (
                progress.loc[
                    i - 1,
                    "EMA Foreground Dice [%]"
                ] * 0.9 + progress.loc[i, "Mean Foreground Dice [%]"] * 0.1    # type: ignore
            )

        progress["Training Loss"] = progress["Training Segmentation Loss"] \
            + (progress["Training CrossEntropy Loss"] - 1)
        progress["Validation Loss"] = progress["Validation Segmentation Loss"] \
            + (progress["Validation CrossEntropy Loss"] - 1)

        return progress[[
            "Epoch",
            "Mean Foreground Dice [%]",
            "EMA Foreground Dice [%]",
            "Training Loss",
            "Validation Loss",
            "Runtime"
        ]]


    def _load_baseline_data(        # noqa: C901, PLR0912
            self,
            datasets: list[str]
        ) -> BaselineResult:
        """Load all baseline data for the given datasets.

        Parameters
        ----------
        datasets : list[str]
            The datasets to load.

        Returns:
        -------
        BaselineResult
            The baseline results.
        """
        all_progress  = []
        all_emissions = []
        all_metrics = []

        for approach, baseline_dir in zip(
            list(APPROACH_REPLACE_MAP.values())[:4],
            [
                self.baseline_conv,
                self.baseline_resenc_m,
                self.baseline_resenc_l,
                self.baseline_medsam2
            ],
            strict=False
        ):
            for dataset in datasets:
                if dataset in self._baseline_datasets:
                    continue

                dataset_progress = []
                dataset_dir = baseline_dir / dataset
                if not dataset_dir.exists():
                    self.logger.info(f"{approach}: Skipping {dataset}.")
                    continue

                for fold in range(self.n_folds):
                    if approach == "MedSAM2":
                        fold_dir = dataset_dir / f"fold_{fold}"
                    else:
                        fold_dir = dataset_dir / self.configuration / f"fold_{fold}"

                        if not (fold_dir / "validation" /\
                                NNUNET_VALIDATION_METRICS_FILENAME).exists():
                            self.logger.info(
                                f"{approach}: Skipping {fold} of dataset {dataset}."
                            )
                            continue

                    if approach == "MedSAM2":
                        progress = self._load_medsam2_progress(fold_dir=fold_dir)
                    else:
                        progress = self._load_nnunet_progress(fold_dir=fold_dir)

                    if (fold_dir / EMISSIONS_FILENAME).is_file():
                        emissions = pd.read_csv(fold_dir / EMISSIONS_FILENAME)
                    else:
                        emissions = pd.DataFrame()

                    if approach == "MedSAM2":
                        metrics = self._load_medsam2_metrics(fold_dir=fold_dir)
                    else:
                        metrics = self._load_nnunet_metrics(fold_dir=fold_dir)

                    for df in [progress, emissions, metrics]:
                        df["Approach"] = approach
                        df["Fold"] = fold
                        df["Dataset"] = dataset

                    all_emissions.append(emissions)
                    all_metrics.append(metrics)
                    dataset_progress.append(progress)

                # We want to calculate the real runtime used by averaging
                # across folds and then summing up the averages
                dataset_progress = pd.concat(dataset_progress) \
                    if dataset_progress else pd.DataFrame()
                if len(dataset_progress) > 0:
                    average_runtime = dataset_progress.groupby(
                        ["Epoch"])["Runtime"].transform("mean")
                    dataset_progress.loc[:, "Average Runtime"] = average_runtime
                    dataset_progress.loc[
                        :, "Real Runtime Used"
                    ] = dataset_progress.groupby(
                        ["Fold"]
                    )["Average Runtime"].cumsum() / 3600

                all_progress.append(dataset_progress)

        all_progress = pd.concat(all_progress)
        all_emissions = pd.concat(all_emissions)
        all_metrics = pd.concat(all_metrics)

        return BaselineResult(
            progress=all_progress,
            metrics=all_metrics,
            emissions=all_emissions
        )

    def _load_incumbent(
            self,
            run_path: Path,
            filename: str = INCUMBENT_FILENAME,
            objective: str = "o0_loss"
        ) -> pd.DataFrame:
        """Load the incumbent logs of the hypersweeper.

        Parameters
        ----------
        run_path : Path
            The run path.

        filename : str
            The filename of the incumbent. Defaults to INCUMBENT_FILENAME.

        objective : str
            The objective to analyze. Defaults to "o0_loss".

        Returns:
        -------
        pd.DataFrame
            The incumbent logs.
        """
        incumbent = pd.read_csv(run_path / filename)
        history = pd.read_csv(run_path / HISTORY_FILENAME)

        # Since we run succesive halving, we have to insert the real
        # budget of a run by subtracting the budget of the previous run
        # in the runhistory (because we use checkpoints)
        history.loc[:, "real_budget"] = 0.
        for run_id, real_budget in self.real_budgets_per_config.items():
            history.loc[history["run_id"] == run_id, "real_budget"] = real_budget

        # For the runtime, we have to add up all indivual runtimes
        # (assuming that we can run 5 folds in parallel)
        history.loc[:, "real_runtime"] = 0.
        for run_id in history["run_id"].unique():
            runtimes = []
            for fold in range(self.n_folds):
                slurm_run_id = run_id * self.n_folds + fold
                progress = self._load_nnunet_progress(run_path / str(slurm_run_id))
                runtimes.append(progress["Runtime"].sum())

            history.loc[
                history["run_id"] == run_id,
                "real_runtime"
            ] = float(np.mean(runtimes)) / 3600

        # The real used budget is the sum of all additional budgets
        incumbent["real_budget_used"] = history["real_budget"].cumsum()

        # Similarly for the runtime
        incumbent["real_runtime_used"] = history["real_runtime"].cumsum()

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
                    O_DSC: performance * 100,
                    "Budget": row["budget"],
                    "Fold": fold,
                    "Budget Used": row["budget_used"],
                    "Real Budget Used": row["real_budget_used"],
                    "Real Runtime Used": row["real_runtime_used"],
                }

                incumbent_expanded.append(row_data)

        incumbent = pd.DataFrame(incumbent_expanded)

        # In addition to the hypersweeper incubment, we also log the
        # config origins manually in NePS. Now we want to merge them
        if origins := self._load_sampling_policy(run_path / SAMPLING_POLICY_LOGS):
            incumbent.loc[:, "Config Origin"] = ""
            for run_id, origin in origins.items():
                incumbent.loc[
                    incumbent["Configuration ID"] == run_id,
                    "Config Origin"
                ] = origin

        incumbent["Full Model Trainings"] = incumbent["Real Budget Used"] /\
            self.max_budget

        return incumbent

    def _load_sampling_policy(self, log_path: Path) -> dict[int, str] | None:
        """Load the sampling policy logs of NePS.

        Parameters
        ----------
        log_path : Path
            The path to the log file.

        Returns:
        -------
        dict[int, str] | None
            The origins of the configurations if available.
        """
        if not log_path.exists():
            return None

        with open(log_path) as file:
            lines = file.readlines()

        lines = [line.strip() for line in lines]

        # In case the optimization was re-run due to some job crashes,
        # we only keep the origins from the last full run
        if len(lines) > 128:        # noqa: PLR2004
            lines = lines[-128:]

        origins = {i + 1: ORIGIN_MAP[origin] for i, origin in enumerate(lines)}
        origins[0] = "Default"

        return origins

    def _load_history(self, run_path: Path) -> pd.DataFrame:
        """Load the runhistory of the hypersweeper.

        Parameters
        ----------
        run_path : Path
            The run path.

        Returns:
        -------
        pd.DataFrame
            The runhistory.
        """
        history = pd.read_csv(run_path / HISTORY_FILENAME)
        history = history.rename(columns=HISTORY_REPLACEMENT_MAP)
        history[O_DSC] *= 100

        # In addition to the hypersweeper history, we also log the
        # config origins manually in NePS. Now we want to merge them
        if origins := self._load_sampling_policy(run_path / SAMPLING_POLICY_LOGS):
            history.loc[:, "Config Origin"] = ""
            for run_id, origin in origins.items():
                history.loc[
                    history["Configuration ID"] == run_id,
                    "Config Origin"
                ] = origin

        return history

    def _load_hpo_data(         # noqa: PLR0915
            self,
            datasets: list[str]
        ) -> HPOResult:
        """Load all HPO data for the given datasets.

        Parameters
        ----------
        datasets : list[str]
            The datasets to load.

        Returns:
        -------
        HPOResult
            The HPO results.
        """
        all_progress  = []
        all_metrics = []
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
                        progress = self._load_nnunet_progress(run_dir)
                        progress["Dataset"] = dataset
                        progress["Approach"] = approach
                        progress["Fold"] = fold
                        progress["Run ID"] = run_id
                        all_progress.append(progress)

                        metrics = self._load_nnunet_metrics(run_dir)
                        metrics["Dataset"] = dataset
                        metrics["Approach"] = approach
                        metrics["Fold"] = fold
                        all_metrics.append(metrics)

        all_progress = pd.concat(all_progress)
        all_metrics = pd.concat(all_metrics)
        all_emissions = pd.concat(all_emissions)
        all_history = pd.concat(all_history)
        all_incumbent = pd.concat(all_incumbent)

        return HPOResult(
            history=all_history,
            incumbent=all_incumbent,
            incumbent_progress=all_progress,
            incumbent_metrics=all_metrics,
            emissions=all_emissions,
            deepcave_runs=deepcave_runs
        )

    def _load_nas_data(     # noqa: PLR0915, PLR0912, C901
            self,
            datasets: list[str],
            approach_key: str
        ) -> NASResult:
        """Load all NAS/HNAS data for the given datasets.

        Parameters
        ----------
        datasets : list[str]
            The datasets to load.

        approach_key : str
            The approach key (hpo_nas, hpo_hnas).

        Returns:
        -------
        NASResult
            The NAS/HNAS results.
        """
        all_emissions = []
        all_history = []
        all_incumbent = defaultdict(list)
        all_metrics = []
        all_progress = []
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

                if i == 0:
                    incumbent_run_id = incumbent["Run ID"].to_numpy()[-1]
                    for fold in range(self.n_folds):
                        metrics = self._load_nnunet_metrics(
                            nas_run_dir / str(incumbent_run_id * self.n_folds + fold)
                        )
                        metrics["Dataset"] = dataset
                        metrics["Approach"] = approach
                        metrics["Fold"] = fold
                        all_metrics.append(metrics)

                        progress = self._load_nnunet_progress(
                            nas_run_dir / str(incumbent_run_id * self.n_folds + fold)
                        )
                        progress["Dataset"] = dataset
                        progress["Approach"] = approach
                        progress["Fold"] = fold
                        all_progress.append(progress)

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

        if len(all_emissions) > 0:
            all_emissions = pd.concat(all_emissions)
            all_history = pd.concat(all_history)
            all_metrics = pd.concat(all_metrics)
            all_progress = pd.concat(all_progress)
        else:
            all_emissions = pd.DataFrame()
            all_history = pd.DataFrame()
            all_metrics = pd.DataFrame()
            all_progress = pd.DataFrame()

        all_incumbent_df = {}
        for objective in self.objectives:
            if len(all_incumbent[objective]) > 0:
                all_incumbent_df[objective] = pd.concat(all_incumbent[objective])
            else:
                all_incumbent_df[objective] = pd.DataFrame()

        return NASResult(
            history=all_history,
            incumbents=all_incumbent_df,
            incumbent_metrics=all_metrics,
            incumbent_progress=all_progress,
            emissions=all_emissions,
            deepcave_runs=deepcave_runs
        )

    def _format_axis(
            self,
            ax: Any,
            grid: bool = False      # noqa: FBT001, FBT002
        ) -> None:
        """Formats a matplotlib axis to match the overall format.

        Parameters
        ----------
        ax : Any
            The axis to format.

        grid : bool
            Whether to add grid lines. Defaults to False.
        """
        major_length = 4
        minor_length = 2
        linewidth = 0.8
        grid_alpha = 0.8

        ax.minorticks_on()
        ax.tick_params(
            axis="x",
            which="major",
            length=major_length,
            width=linewidth,
            color="black"
        )
        ax.tick_params(
            axis="x",
            which="minor",
            length=minor_length,
            width=linewidth,
            color="black"
        )
        ax.tick_params(
            axis="y",
            which="major",
            length=major_length,
            width=linewidth,
            color="black"
        )
        ax.tick_params(
            axis="y",
            which="minor",
            length=minor_length,
            width=linewidth,
            color="black"
        )
        ax.spines["bottom"].set_color("black")
        ax.spines["left"].set_color("black")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        if grid:
            ax.grid(
                which="major",
                linestyle="-",
                linewidth=linewidth,
                alpha=grid_alpha,
                color="lightgray",
                zorder=-1
            )
            ax.grid(
                which="minor",
                linestyle="-",
                linewidth=linewidth,
                alpha=grid_alpha,
                color="lightgray",
                zorder=-1
            )

    def _format_log_xaxis(self, ax: Any) -> None:
        """Formats a matplotlib axis with a logarithmic x-axis scale.

        Parameters
        ----------
        ax : Any
            The axis to format.
        """
        ax.set_xscale("log")

        x_min, x_max = ax.get_xlim()

        log_major_locator = LogLocator(base=10.0, subs=(10,), numticks=2)
        log_minor_locator = LogLocator(base=10.0, subs="auto", numticks=10)
        ax.xaxis.set_major_locator(log_major_locator)
        ax.xaxis.set_minor_locator(log_minor_locator)
        minor_ticks = log_minor_locator.tick_values(x_min, x_max)

        valid_minor_ticks = [
            tick for tick in minor_ticks if x_min <= tick <= x_max]
        closest_minor_tick = min(
            valid_minor_ticks,
            key=lambda t: abs(t - x_min)
        ) if valid_minor_ticks else None

        def minor_formatter(val, _):
            if np.isclose(val, 10) or np.isclose(val, 20) \
                or np.isclose(val, 30) or np.isclose(val, 50):
                return f"{val:.1f}"
            if closest_minor_tick is not None and np.isclose(val, closest_minor_tick):
                if val < 10:   # noqa: PLR2004
                    return f"{val:.2f}"
                return f"{val:.1f}"
            return ""

        ax.xaxis.set_minor_formatter(FuncFormatter(minor_formatter))

        def major_formatter(val, _):
            return f"{val:.1f}"
        ax.xaxis.set_major_formatter(FuncFormatter(major_formatter))

    def _plot_baseline(self, dataset: str, x_metric: str = "Real Runtime Used") -> None:
        """Plot the baseline training progress.

        Parameters
        ----------
        dataset : str
            The dataset to plot.

        x_metric : str
            The metric to use for the x-axis. Defaults to "Real Runtime Used".
        """
        baseline_progress = self.get_baseline_data(dataset).progress

        fig, ax = plt.subplots(1, 1, figsize=(self.figwidth / 2, self.figwidth / 2))

        g = sns.lineplot(
            x=x_metric,
            y="EMA Foreground Dice [%]",
            data=baseline_progress,
            hue="Approach",
            errorbar=("ci", 95),
        )

        g.set_title(f"Training Progress on\n{format_dataset_name(dataset)}")
        if x_metric == "Epoch":
            g.set_xlabel("Epoch")
        elif x_metric == "Real Runtime Used":
            g.set_xlabel("Wallclock Time [h]")

        g.set_ylabel("EMA Mean Foreground DSC (Proxy) [%]")
        g.set_xscale("log")

        g.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.17),
            ncol=2,
            fancybox=False,
            shadow=False,
            frameon=False
        )
        self._format_axis(ax=g, grid=True)

        fig.subplots_adjust(
            top=0.88,
            bottom=0.23,
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
            x_metric: str = "Real Runtime Used"
        ):
        """Plots the baseline training progress for all datasets.

        Parameters
        ----------
        x_metric : str
            The metric to use for the x-axis. Defaults to "Real Runtime Used".
        """
        for dataset in self.datasets:
            self._plot_baseline(dataset=dataset, x_metric=x_metric)

    def _plot_optimization(
            self,
            dataset: str,
            x_log_scale: bool = False,      # noqa: FBT001, FBT002
            y_log_scale: bool = False,      # noqa: FBT001, FBT002
            include_nas: bool = True,       # noqa: FBT001, FBT002
            include_hnas: bool = True,      # noqa: FBT001, FBT002
            show_error: bool = False,       # noqa: FBT001, FBT002
            x_metric: str = "Real Runtime Used"
        ) -> None:
        """Plot the optimization progress of all approaches.

        Parameters
        ----------
        dataset : str
            The dataset to plot.

        x_log_scale : bool
            Whether to use a logarithmic x-axis scale. Defaults to False.

        y_log_scale : bool
            Whether to use a logarithmic y-axis scale. Defaults to False.

        include_nas : bool
            Whether to include the NAS approach. Defaults to True.

        include_hnas : bool
            Whether to include the HNAS approach. Defaults to True.

        show_error : bool
            Whether to show error bars. Defaults to False.

        x_metric : str
            The metric to use for the x-axis. Defaults to "Real Runtime Used".
        """
        color_palette = self.color_palette[:5] + [self.color_palette[9]]

        fig, ax = plt.subplots(1, 1, figsize=(self.figwidth / 2, self.figwidth / 2))

        hpo_data = self.get_hpo_data(dataset)
        incumbent = hpo_data.incumbent

        if include_nas:
            nas_data = self.get_nas_data(dataset)
            incumbent = pd.concat([incumbent, nas_data.incumbents[O_DSC]])
        if include_hnas:
            hnas_data = self.get_hnas_data(dataset)
            incumbent = pd.concat([incumbent, hnas_data.incumbents[O_DSC]])

        baseline_data = self.get_baseline_data(dataset)

        metrics = baseline_data.metrics
        metrics.loc[:, O_DSC] = (100 - metrics["Mean"]).copy()
        metrics = metrics[[O_DSC, "Approach", "Fold"]]

        # We remove the MedSAM2 approach from the baseline
        metrics = metrics[~metrics["Approach"].isin(["MedSAM2"])]

        metrics_expanded = pd.DataFrame(
            np.repeat(metrics.values, 2, axis=0),
            columns=metrics.columns
        )
        metrics_expanded[x_metric] = np.tile(
            [0, incumbent[x_metric].max()],
            len(metrics)
        )

        g = sns.lineplot(
            data=metrics_expanded,
            x=x_metric,
            y=O_DSC,
            hue="Approach",
            palette=self.color_palette[:3],
            linestyle="--",
            errorbar=("sd") if show_error else None,
        )

        n_hpo_approaches = len(incumbent["Approach"].unique())
        sns.lineplot(
            x=x_metric,
            y=O_DSC,
            data=incumbent,
            drawstyle="steps-post",
            hue="Approach",
            errorbar=("sd") if show_error else None,
            palette=color_palette[3: n_hpo_approaches + 3],
            ax=g
        )

        # We add markers to highlight the final values
        hpo_approaches = incumbent["Approach"].unique()
        for i in range(n_hpo_approaches):
            approach = hpo_approaches[i]
            grouped_approach = incumbent[
                incumbent["Approach"] == approach
            ].groupby(x_metric)
            last_value = grouped_approach[O_DSC].mean().iloc[-1]
            last_x = grouped_approach[x_metric].mean().iloc[-1]

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
        if x_metric == "Full Model Trainings":
            g.set_xlabel(x_metric)
        elif x_metric == "Real Runtime Used":
            g.set_xlabel("Wallclock Time [h]")
        g.set_ylabel(O_DSC)

        if x_log_scale:
            g.set_xscale("log")
            if x_metric == "Full Model Trainings":
                g.set_xlim(0.1, self.show_n_full_trainings)
        elif x_metric == "Full Model Trainings":
            g.set_xlim(0, self.show_n_full_trainings)

        if y_log_scale:
            g.set_ylim(1e-2, 1)
            g.set_yscale("log")
        else:
            pass

        self._format_axis(ax=g, grid=True)

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
            left=0.2,
            right=0.94,
        )

        plt.savefig(
            self.combined_plots / f"performance_over_time_{dataset}.{self.format}",
            format=self.format,
            dpi=self.dpi
        )
        plt.clf()

    def plot_optimization(
            self,
            x_log_scale: bool = False,      # noqa: FBT001, FBT002
            y_log_scale: bool = False,      # noqa: FBT001, FBT002
            include_nas: bool = True,       # noqa: FBT001, FBT002
            include_hnas: bool = True,      # noqa: FBT001, FBT002
            show_error: bool = False,       # noqa: FBT001, FBT002
            x_metric: str = "Real Runtime Used"
        ) -> None:
        """Plot the optimization progress of all approaches for all datasets.

        Parameters
        ----------
        x_log_scale : bool
            Whether to use a logarithmic x-axis scale. Defaults to False.

        y_log_scale : bool
            Whether to use a logarithmic y-axis scale. Defaults to False.

        include_nas : bool
            Whether to include the NAS approach. Defaults to True.

        include_hnas : bool
            Whether to include the HNAS approach. Defaults to True.

        show_error : bool
            Whether to show error bars. Defaults to False.

        x_metric : str
            The metric to use for the x-axis. Defaults to "Real Runtime Used".
        """
        for dataset in self.datasets:
            try:
                self._plot_optimization(
                    dataset=dataset,
                    x_log_scale=x_log_scale,
                    y_log_scale=y_log_scale,
                    include_nas=include_nas,
                    include_hnas=include_hnas,
                    show_error=show_error,
                    x_metric=x_metric
                )
            except ValueError:
                self.logger.info(f"Unable to plot HPO for {dataset}.")
                continue

    def plot_optimization_combined(         # noqa: PLR0915, PLR0912, C901
            self,
            x_log_scale: bool = False,      # noqa: FBT001, FBT002
            include_nas: bool = True,       # noqa: FBT001, FBT002
            include_hnas: bool = True,      # noqa: FBT001, FBT002
            show_error: bool = False,       # noqa: FBT001, FBT002
            x_metric: str = "Real Runtime Used"
        ) -> None:
        """Plot the optimization progress of all approaches for all datasets
        in a combined plot.

        Parameters
        ----------
        x_log_scale : bool
            Whether to use a logarithmic x-axis scale. Defaults to False.

        include_nas : bool
            Whether to include the NAS approach. Defaults to True.

        include_hnas : bool
            Whether to include the HNAS approach. Defaults to True.

        show_error : bool
            Whether to show error bars. Defaults to False.

        x_metric : str
            The metric to use for the x-axis. Defaults to "Real Runtime Used".
        """
        color_palette = self.color_palette[:5] + [self.color_palette[9]]

        fig, axes = plt.subplots(
            nrows=2,
            ncols=5,
            figsize=(self.figwidth, 4)
        )
        axes = axes.flatten()

        max_baseline_approaches = 0
        max_hpo_approaches = 0
        max_baseline_approaches_ax = axes[0]
        max_hpo_approaches_ax = axes[0]

        for ax, dataset in zip(axes, self.datasets, strict=False):
            hpo_data = self.get_hpo_data(dataset)
            incumbent = hpo_data.incumbent

            if include_nas and dataset in self.datasets:
                nas_data = self.get_nas_data(dataset)
                if len(nas_data.history) > 0:
                    incumbent = pd.concat([incumbent, nas_data.incumbents[O_DSC]])
            if include_hnas and dataset in self.datasets:
                hnas_data = self.get_hnas_data(dataset)
                if len(hnas_data.history) > 0:
                    incumbent = pd.concat([incumbent, hnas_data.incumbents[O_DSC]])

            baseline_data = self.get_baseline_data(dataset)

            metrics = baseline_data.metrics

            # We remove the MedSAM2 approach from the baseline
            metrics = metrics[~metrics["Approach"].isin(["MedSAM2"])]

            metrics_expanded = pd.DataFrame(
                np.repeat(metrics.values, 2, axis=0),
                columns=metrics.columns
            )
            metrics_expanded[x_metric] = np.tile(
                [0, incumbent[x_metric].max()],
                len(metrics)
            )
            metrics_expanded.loc[:, O_DSC] = (100 - metrics_expanded["Mean"])

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
                x=x_metric,
                y=O_DSC,
                hue="Approach",
                palette=color_palette[:min(3, n_baseline_approaches)],
                linestyle="--",
                errorbar=("sd") if show_error else None,
                ax=ax
            )

            # Plot our approaches
            sns.lineplot(
                x=x_metric,
                y=O_DSC,
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
                grouped_approach = incumbent[
                    incumbent["Approach"] == approach
                ].groupby(x_metric)
                last_value = grouped_approach[O_DSC].mean().iloc[-1]
                last_x = grouped_approach[x_metric].mean().iloc[-1]

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
            if x_metric == "Full Model Trainings":
                g.set_xlabel(x_metric)
            elif x_metric == "Real Runtime Used":
                g.set_xlabel("Wallclock Time [h]")
            g.set_ylabel(O_DSC)

            if x_log_scale:
                g.set_xscale("log")
                if x_metric == "Full Model Trainings":
                    g.set_xlim(0.1, self.show_n_full_trainings)
            elif x_metric == "Full Model Trainings":
                g.set_xlim(0, self.show_n_full_trainings)

            if ax != axes[0] and ax != axes[5]:
                ax.set_ylabel("")

            if ax != axes[5] and ax != axes[6] and ax != axes[7] \
                and ax != axes[8] and ax != axes[9]:
                ax.set_xlabel("")

            ax.get_legend().remove()
            ax.yaxis.set_tick_params(pad=1)

        # We use the axis with most approaches to get the legend
        (
            baseline_handles,
            baseline_labels
        ) = max_baseline_approaches_ax.get_legend_handles_labels()
        (
            hpo_handles,
            hpo_labels
        ) = max_hpo_approaches_ax.get_legend_handles_labels()

        zipped_handles = [
            val for pair in zip(
                baseline_handles[:max_baseline_approaches],
                hpo_handles[-max_hpo_approaches:],
                strict=False
            ) for val in pair
        ]
        zipped_labels = [
            val for pair in zip(
                baseline_labels[:max_baseline_approaches],
                hpo_labels[-max_hpo_approaches:],
                strict=False
            ) for val in pair
        ]

        fig.subplots_adjust(
            top=0.93,
            bottom=0.24,
            left=0.09,
            right=0.975,
            hspace=0.45,
            wspace=0.49
        )

        axes[-3].legend(
            zipped_handles,
            zipped_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.4),
            ncol=3,
            fancybox=False,
            shadow=False,
            frameon=False
        )

        plt.savefig(
            self.combined_plots / f"performance_over_time.{self.format}",
            format=self.format,
            dpi=self.dpi
        )
        plt.clf()

    def _plot_nas_combined(
            self,
            dataset: str,
        ) -> None:
        """Plot the Pareto fronts of all approaches for a given dataset.

        Parameters
        ----------
        dataset : str
            The dataset to plot.
        """
        fig, ax = plt.subplots(1, 1, figsize=(self.figwidth / 2, self.figwidth / 2))

        baseline_data = self.get_baseline_data(dataset)
        n_baseline_approaches = 0

        for baseline_approach in baseline_data.metrics["Approach"].unique():
            metrics = baseline_data.metrics
            metrics = metrics[metrics["Approach"] == baseline_approach]
            baseline_dice = metrics["Mean"].mean()

            baseline_progress = baseline_data.progress
            baseline_progress = baseline_progress[
                baseline_progress["Approach"] == baseline_approach
            ]
            baseline_time = baseline_progress.groupby(
                "Fold")["Runtime"].sum().mean() / 3600

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

        # get mean of last 5 entries in incumbent
        hpo_data = self.get_hpo_data(dataset)
        hpo_dice = hpo_data.incumbent[O_DSC].iloc[-5:].mean()
        hpo_time = hpo_data.incumbent_progress.groupby(
            "Fold")["Runtime"].sum().mean() / 3600

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

        nas_data = self.get_nas_data(dataset)
        if len(nas_data.history) > 0:
            nas_history = nas_data.history
            nas_pareto_front = nas_history.sort_values(by=O_DSC)
            nas_pareto_front = nas_pareto_front[
                nas_pareto_front[O_RUNTIME] == nas_pareto_front[O_RUNTIME].cummin()
            ]

            sns.lineplot(
                data=nas_pareto_front,
                x=O_DSC,
                y=O_RUNTIME,
                color=self.color_palette[n_baseline_approaches + 1],
                label="HPO + NAS (ours)",
                ax=ax,
                drawstyle="steps-post",
                zorder=5
            )

        hnas_data = self.get_hnas_data(dataset)
        if len(hnas_data.history) > 0:
            hnas_history = hnas_data.history
            hnas_pareto_front = hnas_history.sort_values(by=O_DSC)
            hnas_pareto_front = hnas_pareto_front[
                hnas_pareto_front[O_RUNTIME] == hnas_pareto_front[O_RUNTIME].cummin()
            ]

            sns.lineplot(
                data=hnas_pareto_front,
                x=O_DSC,
                y=O_RUNTIME,
                color=self.color_palette[9],
                label="HPO + HNAS (ours)",
                ax=ax,
                drawstyle="steps-post",
                zorder=5
            )
        ax.set_title(f"Pareto Fronts for\n{format_dataset_name(dataset)}")
        ax.set_xlabel(O_DSC)
        ax.set_ylabel("Training Runtime [h]")

        self._format_axis(ax=ax, grid=True)
        self._format_log_xaxis(ax=ax)
        ax.set_yscale("log")

        fig.subplots_adjust(
            top=0.89,
            bottom=0.32,
            left=0.17,
            right=0.94,
        )

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            fancybox=False,
            shadow=False,
            frameon=False
        )

        plt.savefig(
            self.combined_plots / f"pareto_fronts_{dataset}.{self.format}",
            format=self.format,
            dpi=self.dpi
        )
        plt.clf()

    def plot_nas_combined(self) -> None:
        """Plot the Pareto fronts of all approaches for all datasets."""
        for dataset in self.datasets:
            self._plot_nas_combined(dataset)

    def _plot_nas_budgets(
            self,
            dataset: str,
            approach_key: str,
        ) -> None:
        """Plot the budgets of the NAS/HNAS approaches.

        Parameters
        ----------
        dataset : str
            The dataset to plot.

        approach_key : str
            The approach key (hpo_nas, hpo_hnas).
        """
        if approach_key == "hpo_nas":
            nas_data = self.get_nas_data(dataset)
        elif approach_key == "hpo_hnas":
            nas_data = self.get_hnas_data(dataset)
        else:
            raise ValueError(f"Unknown approach key {approach_key}.")

        baseline_data = self.get_baseline_data(dataset)

        fig, ax = plt.subplots(1, 1, figsize=(self.figwidth / 2, self.figwidth / 2))

        n_baseline_approaches = 0

        for baseline_approach in baseline_data.metrics["Approach"].unique():
            metrics = baseline_data.metrics
            metrics = metrics[metrics["Approach"] == baseline_approach]
            baseline_dice = metrics["Mean"].mean()

            baseline_progress = baseline_data.progress
            baseline_progress = baseline_progress[
                baseline_progress["Approach"] == baseline_approach
            ]
            baseline_time = baseline_progress.groupby(
                "Fold")["Runtime"].sum().mean() / 3600

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
        hpo_dice = hpo_data.incumbent[O_DSC].iloc[-5:].mean()
        hpo_time = hpo_data.incumbent_progress.groupby(
            "Fold")["Runtime"].sum().mean() / 3600

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
        pareto_front = history.sort_values(by=O_DSC)
        pareto_front = pareto_front[
            pareto_front[O_RUNTIME] == pareto_front[O_RUNTIME].cummin()
        ]

        g = sns.lineplot(
            data=pareto_front,
            x=O_DSC,
            y=O_RUNTIME,
            color=self.color_palette[9],
            label="HPO + NAS (ours)",
            ax=ax,
            drawstyle="steps-post",
            zorder=5
        )

        # Round budget column
        history.loc[:, "Budget"] = history["Budget"].round()
        g = sns.scatterplot(
            data=history,
            x=O_DSC,
            y=O_RUNTIME,
            # label="Configurations",
            color=self.color_palette[n_baseline_approaches + 1],
            size="Budget",
            s=10,
            alpha=0.5,
            ax=ax,
        )

        approach = APPROACH_REPLACE_MAP[approach_key].replace(" (ours)", "")
        g.set_title(f"{approach} Configuration Budgets for"
                    f"\n{format_dataset_name(dataset)}")
        g.set_xlabel(O_DSC)
        g.set_ylabel("Training Runtime [h]")

        g.set_yscale("log")

        self._format_log_xaxis(ax=ax)
        self._format_axis(ax=g, grid=True)

        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=2,
            fancybox=False,
            shadow=False,
            frameon=False
        )

        fig.subplots_adjust(
            top=0.89,
            bottom=0.415,
            left=0.17,
            right=0.97,
        )

        output_dir = self.nas_plots if approach_key == "hpo_nas" else self.hnas_plots
        plt.savefig(
            output_dir / f"{dataset}_budgets.{self.format}",
            format=self.format,
            dpi=self.dpi
        )
        plt.clf()

    def plot_nas_budgets(self, approach_key: str) -> None:
        """Plot the budgets of the NAS/HNAS approaches for all datasets.

        Parameters
        ----------
        approach_key : str
            The approach key (hpo_nas, hpo_hnas).
        """
        for dataset in self.datasets:
            self._plot_nas_budgets(dataset, approach_key=approach_key)

    def _plot_nas_origins(
            self,
            dataset: str,
            approach_key: str,
        ) -> None:
        """Plot the origins of the NAS/HNAS approaches.

        Parameters
        ----------
        dataset : str
            The dataset to plot.

        approach_key : str
            The approach key (hpo_nas, hpo_hnas).
        """
        if approach_key == "hpo_nas":
            nas_data = self.get_nas_data(dataset)
        elif approach_key == "hpo_hnas":
            nas_data = self.get_hnas_data(dataset)
        else:
            raise ValueError(f"Unknown approach key {approach_key}.")

        fig, ax = plt.subplots(1, 1, figsize=(self.figwidth / 2, self.figwidth / 2))

        history = nas_data.history

        pareto_front = history.sort_values(by=O_DSC)
        pareto_front = pareto_front[
            pareto_front[O_RUNTIME] == pareto_front[O_RUNTIME].cummin()
        ]

        g = sns.lineplot(
            data=pareto_front,
            x=O_DSC,
            y=O_RUNTIME,
            color=self.color_palette[0],
            label="Pareto Front",
            ax=ax,
            drawstyle="steps-post",
            zorder=5
        )

        origins = [
            origin for origin in history["Config Origin"].unique() \
                if origin != "Default"
        ]
        markers = ["x", "x", "x"]
        origin_markers = dict(zip(origins, markers, strict=False))
        colors = self.color_palette[1:4]
        origin_colors = dict(zip(origins, colors, strict=False))

        for origin in origins:
            subset = history[history["Config Origin"] == origin]
            marker = origin_markers[origin]
            color = origin_colors[origin]
            g = sns.scatterplot(
                data=subset,
                x=O_DSC,
                y=O_RUNTIME,
                label=origin,
                color=color,
                marker=marker,
                ax=ax,
                linewidth=1.25,
                alpha=0.8,
            )

        approach = APPROACH_REPLACE_MAP[approach_key].replace(" (ours)", "")
        g.set_title(f"{approach} Configuration Origins for"
                    f"\n{format_dataset_name(dataset)}")
        g.set_xlabel(O_DSC)
        g.set_ylabel("Training Runtime [h]")

        self._format_axis(ax=g, grid=True)
        self._format_log_xaxis(ax=ax)

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
            top=0.88,
            bottom=0.25,
            left=0.17,
            right=0.97,
        )

        output_dir = self.nas_plots if approach_key == "hpo_nas" else self.hnas_plots
        plt.savefig(
            output_dir / f"{dataset}_origins.{self.format}",
            format=self.format,
            dpi=self.dpi
        )
        plt.clf()
        plt.close()

    def plot_nas_origins(self, approach_key: str) -> None:
        """Plot the origins of the NAS/HNAS approaches for all datasets.

        Parameters
        ----------
        approach_key : str
            The approach key (hpo_nas, hpo_hnas).
        """
        for dataset in self.datasets:
            self._plot_nas_origins(dataset, approach_key=approach_key)

    @staticmethod
    def _get_budget(budget: int, deepcave_run: DeepCAVERun) -> float | int:
        """Get the actual budget for a given index.

        Parameters
        ----------
        budget : int
            The budget index.

        deepcave_run : DeepCAVERun
            The DeepCAVE run object.

        Returns:
        -------
        float | int
            The actual budget.
        """
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
            method: Literal["global", "local"] = "global",
            plot_pdps: bool = False     # noqa: FBT001, FBT002
        ) -> None:
        """Plot the hyperparameter importances for all datasets.

        Parameters
        ----------
        budget : int
            The budget to use. Defaults to COMBINED_BUDGET.

        method : str
            The method to use. Either "global" or "local". Defaults to "global".

        plot_pdps : bool
            Whether to plot the PDPs for the most important hyperparameter per Dataset.
            Defaults to False.
        """
        pdps = []

        fig, axes = plt.subplots(
            nrows=2,
            ncols=5,
            sharex=True,
            figsize=(self.figwidth, 5)
        )
        axes = axes.flatten()

        hyperparameters = {
            k: v for k, v in HYPERPARAMETER_REPLACEMENT_MAP.items() \
                if k in HPO_HYPERPARAMETERS
        }

        for ax, dataset in zip(axes, self.datasets, strict=False):
            deepcave_run = self.get_hpo_data(dataset).deepcave_runs[dataset]

            selected_budget = self._get_budget(budget, deepcave_run)

            if method == "global":
                evaluator = fANOVA(run=deepcave_run)
                evaluator.calculate(budget=selected_budget, seed=42)
            else:
                evaluator = LPI(run=deepcave_run)
                evaluator.calculate(budget=selected_budget, seed=42)

            importances = evaluator.get_importances(
                hp_names=list(deepcave_run.configspace.keys())
            )

            importances_data = []
            for hp_key, hp_name in hyperparameters.items():
                importance = importances[hp_key]
                importances_data += [{
                    "Hyperparameter": hp_name,
                    "Importance": importance[0],
                    "Error": importance[1],
                }]
            importances_df = pd.DataFrame(importances_data)

            if plot_pdps and method == "global":
                most_important_hp = importances_df.loc[
                    importances_df["Importance"].idxmax()
                ]["Hyperparameter"]
                assert isinstance(most_important_hp, str)

                pdps += [(dataset, most_important_hp)]

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

            ax.set_yticks([])
            ax.tick_params(axis="y", which="both", left=False)

            self._format_axis(ax=g, grid=True)

            if ax == axes[0] or ax == axes[5]:
                g.set_ylabel("Hyperparameter")
            else:
                g.set_ylabel("")

        axes[-1].legend().set_visible(False)

        fig.subplots_adjust(
            top=0.85,
            bottom=0.17,
            left=0.04,
            right=0.98,
            hspace=0.32,
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

        if len(pdps) > 0:
            for dataset, hp in pdps:
                self.plot_pdp(
                    dataset=dataset,
                    approach_key="hpo",
                    hp_name_1=hp,
                )

    def _plot_mo_hp_importances(        # noqa: PLR0915, PLR0912, C901
            self,
            dataset: str,
            approach_key: str,
            budget: int = COMBINED_BUDGET,
            method: Literal["global", "local"] = "global",
            plot_pdps: bool = False     # noqa: FBT001, FBT002
        ) -> None:
        """Plot the multi-objective hyperparameter importances for a given dataset.

        Parameters
        ----------
        dataset : str
            The dataset to plot.

        approach_key : str
            The approach key (hpo_nas, hpo_hnas).

        budget : int
            The budget to use. Defaults to COMBINED_BUDGET.

        method : str
            The method to use. Either "global" or "local". Defaults to "global".

        plot_pdps : bool
            Whether to plot the PDPs for the most important hyperparameters per Dataset.
            Defaults to False.
        """
        pdps = []

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
        nas_hp_names = NAS_HYPERPARAMETERS \
            if approach_key == "hpo_nas" else HNAS_HYPERPARAMETERS

        fig_height = 4.5 if approach_key == "hpo_hnas" else 4

        fig, axs = plt.subplots(
            nrows=1,
            ncols=2,
            sharex=True,
            sharey=True,
            figsize=(self.figwidth, fig_height)
        )

        assert all_importances is not None
        if len(all_importances) == 0:
            return

        for i, ax in enumerate(axs):
            if i == 0:
                # HPO
                importances = all_importances[
                    all_importances["hp_name"].isin(HPO_HYPERPARAMETERS)
                ]
                importances.loc[:, "hp_name"] = importances.loc[:, "hp_name"].replace(
                    HYPERPARAMETER_REPLACEMENT_MAP
                )
                hp_names = [
                    v for k, v in HYPERPARAMETER_REPLACEMENT_MAP.items() \
                        if k in HPO_HYPERPARAMETERS
                    ]
            else:
                # NAS or HNAS
                importances = all_importances[
                    all_importances["hp_name"].isin(nas_hp_names)
                ]
                importances.loc[:, "hp_name"] = importances.loc[:, "hp_name"].replace(
                    HYPERPARAMETER_REPLACEMENT_MAP
                )
                hp_names = [
                    v for k, v in HYPERPARAMETER_REPLACEMENT_MAP.items() \
                        if k in nas_hp_names
                ]

                if plot_pdps and method == "global":
                    for weight in [0, 1]:
                        _importances = importances[importances["weight"] == weight]

                        most_important_hp = _importances.loc[
                            _importances["importance"].idxmax()
                        ]["hp_name"]
                        assert isinstance(most_important_hp, str)

                        pdps += [(1 - weight, most_important_hp)]

            for j, hp in enumerate(hp_names):
                hp_data = importances[importances["hp_name"] == hp]
                hp_data = hp_data.sort_values(by="weight")
                x = hp_data["weight"]
                y = hp_data["importance"]
                variance = hp_data["variance"]

                color = self.color_palette[j % len(self.color_palette)]
                linestyle = "--" if j >= len(self.color_palette) else "-"

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
            ax.set_xlabel(f"Weight of {O_DSC}")
            ax.set_ylabel("Importance")

            self._format_axis(ax=ax, grid=True)

        axs[0].set_title("HPO Hyperparameters")
        axs[1].set_title(f"{nas_approach} Hyperparameters")

        # To imitate DeepCAVE, we add the objectives to the xticks
        xticks = ax.get_xticks()
        xticklabels = ax.get_xticklabels()
        xticklabels[1] = "0.0\nRuntime"
        xticklabels[-2] = f"1.0\n{O_DSC}"
        ax.set_xticks(xticks[1:-1])
        ax.set_xticklabels(xticklabels[1:-1])

        # As NAS and HNAS have different numbers of hyperparameters, we need to adjust
        # the layout
        if approach_key == "hpo_hnas":
            fig.subplots_adjust(
                top=0.87,
                bottom=0.49,
                left=0.08,
                right=0.96,
                wspace=0.2,
                hspace=0.5,
            )
        else:
            fig.subplots_adjust(
                top=0.86,
                bottom=0.43,
                left=0.07,
                right=0.96,
                wspace=0.2,
                hspace=0.5,
            )

        _method = "Global" if method == "global" else "Local"
        fig.suptitle(f"{_method} MO-HPIs for HPO + {nas_approach} on "
                     f"{format_dataset_name(dataset)}")
        plt.savefig(
            self.analysis_plots[approach_key] /\
                f"{method}_mo_hpi_{dataset}.{self.format}",
            format=self.format,
            dpi=self.dpi
        )
        plt.clf()

        if len(pdps) > 0:
            for objective_id, hp in pdps:
                self.plot_pdp(
                    dataset=dataset,
                    approach_key=approach_key,
                    hp_name_1=hp,
                    objective_id=objective_id
                )

    def plot_hpis(
            self,
            approach_keys: list[str] | None = None,
            budget: int = COMBINED_BUDGET,
            plot_pdps: bool = False         # noqa: FBT001, FBT002
        ) -> None:
        """Plot the hyperparameter importances for all datasets.

        Parameters
        ----------
        approach_keys : list[str] | None
            The approach keys to use. Defaults to None. In that case, all approaches are
            used.

        budget : int
            The budget to use. Defaults to COMBINED_BUDGET.

        plot_pdps : bool
            Whether to plot the PDPs for the most important hyperparameter per Dataset.
            Defaults to False.
        """
        if approach_keys is None:
            approach_keys = APPROACHES
        for approach_key in approach_keys:
            if approach_key == "hpo":
                self._plot_hp_importances(
                    budget=budget,
                    method="global",
                    plot_pdps=plot_pdps
                )
                self._plot_hp_importances(
                    budget=budget,
                    method="local",
                    plot_pdps=plot_pdps
                )
            else:
                for dataset in self.datasets:
                    self._plot_mo_hp_importances(
                        dataset=dataset,
                        approach_key=approach_key,
                        budget=budget,
                        method="global",
                        plot_pdps=plot_pdps
                    )
                    self._plot_mo_hp_importances(
                        dataset=dataset,
                        approach_key=approach_key,
                        budget=budget,
                        method="local",
                        plot_pdps=plot_pdps
                    )

    def _plot_single_objective_ablation(
            self,
            performances: dict,
            improvements: dict
        ) -> matplotlib.figure.Figure:
        """Plot the single-objective ablation path from the default to incumbent
        configuration.

        Parameters
        ----------
        performances : dict
            The performances of the default and incumbent configurations.

        improvements : dict
            The improvements of the incumbent configuration over the default
            configuration.

        Returns:
        -------
        matplotlib.figure.Figure
            The resulting figure.
        """
        fig, axs = plt.subplots(1, 2, figsize=(self.figwidth, self.figwidth / 2))

        # 1) Performances
        hps = [HYPERPARAMETER_REPLACEMENT_MAP[hp] for hp in performances]

        perf_values = list(performances.values())
        impr_values = list(improvements.values())
        sns.lineplot(
            x=hps,
            y=[mean for mean, _ in perf_values],
            ax=axs[0],
            color=self.color_palette[0]
        )

        for hp, (mean, var) in zip(hps, perf_values, strict=False):
            axs[0].errorbar(
                x=[hp],
                y=[mean],
                yerr=[var / 100],
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
        for hp, (mean, var) in zip(hps[1:], impr_values[1:], strict=False):
            axs[1].errorbar(
                x=[hp],
                y=[mean],
                yerr=[var],
                capsize=5,
                color="black"
            )

        # rotate x-axis labels
        for ax in axs:
            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=45,
                horizontalalignment="right"
            )
            self._format_axis(ax=ax)

        axs[0].set_title("Performance")
        axs[1].set_title("Improvement")

        axs[0].set_ylabel(O_DSC)
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
            importances: pd.DataFrame,
        ) -> matplotlib.figure.Figure:
        """Plot the multi-objective ablation path from the default to incumbent
        configuration.

        Parameters
        ----------
        importances : pd.DataFrame
            The importances of the hyperparameters.

        Returns:
        -------
        matplotlib.figure.Figure
            The resulting figure.
        """
        fig, ax = plt.subplots(1, 1, figsize=(self.figwidth, self.figwidth / 2))


        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            sharex=True,
            figsize=(self.figwidth / 2, self.figwidth / 2)
        )

        importances.loc[:, "hp_name"] = importances.loc[:, "hp_name"].replace(
            HYPERPARAMETER_REPLACEMENT_MAP
        )
        importances.loc[
            importances["hp_name"] == "Default", "accuracy"
        ] = 1 - importances["new_performance"]
        importances.loc[
            importances["hp_name"] != "Default", "accuracy"
        ] = importances["importance"]

        grouped_importances = importances.groupby(      # noqa: PD010
            ["weight", "hp_name"]
        )["accuracy"].sum().unstack(fill_value=0)

        colors = {
            column: self.color_palette[i] \
                for i, column in enumerate(grouped_importances.columns)
        }

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

        ax.set_xlabel(f"Weight of {O_DSC}")
        ax.set_ylabel("Sum of weighted\nnormalized performance")

        # To imitate DeepCAVE, we add the objectives to the xticks
        xticks = ax.get_xticks()
        xticklabels = ax.get_xticklabels()
        xticklabels[1] = "0.0\nRuntime"
        xticklabels[-2] = f"1.0\n{O_DSC}"
        ax.set_xticks(xticks[1:-1])
        ax.set_xticklabels(xticklabels[1:-1])

        self._format_axis(ax=ax)

        fig.subplots_adjust(
            top=0.86,
            bottom=0.34,
            left=0.18,
            right=0.87,
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
    ) -> None:
        """Plot the ablation path for a given dataset and approach.

        Parameters
        ----------
        dataset : str
            The dataset to plot.

        approach_key : str
            The approach key (hpo, hpo_nas, hpo_hnas).

        budget_idx : int
            The budget index to use. Defaults to COMBINED_BUDGET.

        n_hps : int
            The number of hyperparameters to plot. Defaults to 3.
        """
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
            for _key, value in improvements.items():
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
            idx = [*list(idx[:n_hps]), "Default"]
            importances = data[
                data["hp_name"].isin(idx)
            ].copy()

            fig = self._plot_multi_objective_ablation(
                importances=importances,
            )
        _approach = APPROACH_REPLACE_MAP[approach_key].replace(" (ours)", "")
        if approach_key == "hpo":
            fig.suptitle(f"Ablation Path for {_approach} on "
                         f"{format_dataset_name(dataset)}")
        else:
            fig.suptitle(f"MO Ablation Path for {_approach} on"
                         f"\n{format_dataset_name(dataset)}")
        plt.savefig(
            self.analysis_plots[approach_key] / f"ablation_{dataset}.{self.format}",
            format=self.format,
            dpi=self.dpi
        )

    def plot_ablation_paths(
        self,
        approach_keys: list[str] | None = None,
        budget_idx: int = COMBINED_BUDGET,
        n_hps: int = 3
    ) -> None:
        """Plot the ablation paths for all datasets and approaches.

        Parameters
        ----------
        approach_keys : list[str] | None
            The approach keys to use. Defaults to None. In that case, all approaches are
            used.

        budget_idx : int
            The budget index to use. Defaults to COMBINED_BUDGET.

        n_hps : int
            The number of hyperparameters to plot. Defaults to 3.
        """
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

    def _plot_pdp_1hp(
            self,
            deepcave_run: DeepCAVERun,
            outputs: dict,
            hp_name: str,
            objective_id: int = 0,
            incumbent_value: str | int | float | None = None
        ) -> matplotlib.figure.Figure:
        """Plot a partial dependence plot for a single hyperparameter.

        Parameters
        ----------
        deepcave_run : DeepCAVERun
            The DeepCAVE run object.

        outputs : dict
            The outputs of the DeepCAVE PDP evaluator.

        hp_name : str
            The hyperparameter name.

        objective_id : int
            The objective ID. Defaults to 0.

        Returns:
        -------
        matplotlib.figure.Figure
            The resulting figure.
        """
        x = np.asarray(outputs["x"])
        y = np.asarray(outputs["y"])

        objective = deepcave_run.get_objective(objective_id)
        assert objective is not None

        # To account for the objective scale in the plot we
        # normalize the variances by the maximum value of the
        # objective
        objective_max = deepcave_run.get_encoded_data(
            objective,
            COMBINED_BUDGET,
            specific=True,
        )[objective.name].to_numpy().max()

        sigmas = np.sqrt(
            np.asarray(outputs["variances"])
        ) / objective_max
        x_ice = np.asarray(outputs["x_ice"])
        y_ice = np.asarray(outputs["y_ice"])

        hp_idx = deepcave_run.configspace.index_of[hp_name]
        hp = deepcave_run.configspace[hp_name]

        fig, ax = plt.subplots(1, 1, figsize=(self.figwidth / 2, 3))

        for x_, y_ in zip(x_ice, y_ice, strict=False):
            sns.lineplot(
                x=x_[:, hp_idx],
                y=y_,
                ax=ax,
                color=self.color_palette[3],
                alpha=0.1,
                label="ICE"
            )

        sns.lineplot(
            x=x[:, hp_idx],
            y=y,
            ax=ax,
            color=self.color_palette[0],
            linewidth=2,
            label="PDP"
        )

        ax.fill_between(
            x[:, hp_idx],
            y1=y - max(sigmas) * sigmas,
            y2=y + max(sigmas) * sigmas,
            alpha=0.2,
            color=self.color_palette[0],
        )

        tickvals, ticktext = get_hyperparameter_ticks(hp=hp)

        tickvals = tickvals[:-1]
        ticktext = ticktext[:-1]

        ticktext = [HYPERPARAMETER_VALUE_REPLACE_MAP.get(t, t) for t in ticktext]

        if isinstance(hp, CategoricalHyperparameter):
            default_vector = 0
            if incumbent_value is None:
                inc_vector = None
            else:
                inc_idx = np.where(
                    np.array(list(hp.choices)) == incumbent_value
                )[0][0]
                inc_vector = inc_idx / (len(hp.choices) - 1)
        else:
            default_vector = hp.to_vector(hp.default_value)

            if incumbent_value is None:
                inc_vector = None
            else:
                inc_vector = hp.to_vector(incumbent_value)

        # We add vertical lines for the default and incumbent
        ax.axvline(
            x=default_vector,
            color=self.color_palette[2],
            linestyle="-",
            label="Default"
        )
        if inc_vector is not None:
            ax.axvline(
                x=inc_vector,
                color=self.color_palette[3],
                linestyle="--",
                label="Inc."
            )

        ax.set_xticks(tickvals)
        ax.set_xticklabels(ticktext)
        ax.set_xlabel(HYPERPARAMETER_REPLACEMENT_MAP[hp_name])
        ax.set_ylabel(self.objectives[objective_id])

        self._format_axis(ax=ax)

        ax.xaxis.set_minor_locator(plt.NullLocator())

        fig.subplots_adjust(
            top=0.85,
            bottom=0.21,
            left=0.18,
            right=0.9,
            wspace=0.33,
            hspace=0.33
        )

        handles, labels = ax.get_legend_handles_labels()

        if inc_vector is None:
            handles, labels = handles[-3:], labels[-3:]
        else:
            handles, labels = handles[-4:], labels[-4:]

        # We need to swap the ICE and PDP labels
        labels[0], labels[1] = labels[1], labels[0]
        handles[0], handles[1] = handles[1], handles[0]

        ax.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=len(handles),
            fancybox=False,
            shadow=False,
            frameon=False
        )
        return fig

    def _plot_pdp_2hps(
            self,
            deepcave_run: DeepCAVERun,
            outputs: dict,
            hp_name_1: str,
            hp_name_2: str,
            objective_id: int = 0
        ) -> matplotlib.figure.Figure:
        """Plot a partial dependence plot for two hyperparameters.

        Parameters
        ----------
        deepcave_run : DeepCAVERun
            The DeepCAVE run object.


        outputs : dict
            The outputs of the DeepCAVE PDP evaluator.

        hp_name_1 : str
            The first hyperparameter name.

        hp_name_2 : str
            The second hyperparameter name.

        objective_id : int
            The objective ID. Defaults to 0.

        Returns:
        -------
        matplotlib.figure.Figure
            The resulting figure.
        """
        x = np.asarray(outputs["x"])
        y = np.asarray(outputs["y"])
        np.sqrt(np.asarray(outputs["variances"]))
        np.asarray(outputs["x_ice"])
        np.asarray(outputs["y_ice"])

        hp1_idx = deepcave_run.configspace.index_of[hp_name_1]
        hp1 = deepcave_run.configspace[hp_name_1]
        hp2_idx = deepcave_run.configspace.index_of[hp_name_2]
        hp2 = deepcave_run.configspace[hp_name_2]

        fig, ax = plt.subplots(1, 1, figsize=(self.figwidth / 2, 3))

        x_hp1 = x[:, hp1_idx]
        x_hp2 = x[:, hp2_idx]

        contour = plt.tricontourf(
            x_hp1,
            x_hp2,
            y,
            levels=15,
            cmap="plasma",
            alpha=1
        )
        cbar = plt.colorbar(contour)
        cbar.set_label(self.objectives[objective_id])

        xtickvals, xticktext = get_hyperparameter_ticks(hp1)
        ytickvals, yticktext = get_hyperparameter_ticks(hp2)
        ax.set_xticks(xtickvals[:-1])
        ax.set_xticklabels(xticktext[:-1])
        ax.set_yticks(ytickvals[:-1])
        ax.set_yticklabels(yticktext[:-1])
        ax.set_xlabel(HYPERPARAMETER_REPLACEMENT_MAP[hp_name_1])
        ax.set_ylabel(HYPERPARAMETER_REPLACEMENT_MAP[hp_name_2])

        self._format_axis(ax=ax)

        fig.subplots_adjust(
            top=0.89,
            bottom=0.15,
            left=0.2,
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
    ) -> None:
        """Plot a partial dependence plot for a given dataset and approach.

        Parameters
        ----------
        dataset : str
            The dataset to plot.

        approach_key : str
            The approach key (hpo, hpo_nas, hpo_hnas).

        hp_name_1 : str
            The first hyperparameter name.

        hp_name_2 : str | None
            The second hyperparameter name. Defaults to None.

        budget_id : int
            The budget ID. Defaults to COMBINED_BUDGET.

        objective_id : int
            The objective ID. Defaults to 0.
        """
        try:
            if approach_key == "hpo":
                deepcave_run =  self.get_hpo_data(dataset).deepcave_runs[dataset]
            elif approach_key == "hpo_nas":
                deepcave_run = self.get_nas_data(dataset).deepcave_runs[dataset]
            elif approach_key == "hpo_hnas":
                deepcave_run = self.get_hnas_data(dataset).deepcave_runs[dataset]
            else:
                raise ValueError(f"Unknown approach key {approach_key}.")
        except KeyError as e:
            self.logger.error(e)
            return

        objective = deepcave_run.get_objective(0)
        inc_config = deepcave_run.get_incumbent(objectives=objective)[0]

        if hp_name_1 not in HYPERPARAMETER_REPLACEMENT_MAP:
            hp_name_1 = next(
                k for k, v in HYPERPARAMETER_REPLACEMENT_MAP.items() if v == hp_name_1
            )
            inc_1 = inc_config.get(hp_name_1, None)

        if hp_name_2 and hp_name_2 in HYPERPARAMETER_REPLACEMENT_MAP:
            hp_name_2 = next(
                k for k, v in HYPERPARAMETER_REPLACEMENT_MAP.items() if v == hp_name_2
            )

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
            self._plot_pdp_1hp(
                deepcave_run=deepcave_run,
                outputs=outputs,
                hp_name=hp_name_1,
                objective_id=objective_id,
                incumbent_value=inc_1
            )
        else:
            self._plot_pdp_2hps(
                deepcave_run=deepcave_run,
                outputs=outputs,
                hp_name_1=hp_name_1,
                hp_name_2=hp_name_2,
                objective_id=objective_id
            )

        plt.title(f"PDP for {APPROACH_REPLACE_MAP[approach_key].replace(' (ours)', '')}"
                  f" on\n{format_dataset_name(dataset)}")

        if hp_name_2 is None:
            title = f"pdp_{dataset}_objective_{objective_id}"
        else:
            title = f"pdp_{dataset}_{hp_name_1}_{hp_name_2}_objective_{objective_id}"

        plt.savefig(
            self.analysis_plots[approach_key] / f"{title}.{self.format}",
            format=self.format,
            dpi=self.dpi
        )

    def _plot_footprint(        # noqa: PLR0915
            self,
            dataset: str,
            approach_key: str,
            objective: str = O_DSC,
            budget: int = COMBINED_BUDGET
        ) -> None:
        """Plot the configuartion footprint for a given dataset and approach.

        Parameters
        ----------
        dataset : str
            The dataset to plot.

        approach_key : str
            The approach key (hpo, hpo_nas, hpo_hnas).

        objective : str
            The objective to use. Defaults to O_DSC.

        budget : int
            The budget to use. Defaults to COMBINED_BUDGET.
        """
        try:
            deepcave_run, history, incumbent = self.get_deepcave_data(
                dataset=dataset,
                approach_key=approach_key,
                objective=objective
            )
        except KeyError as e:
            self.logger.error(e)
            return

        fp = Footprint(
            run=deepcave_run
        )

        selected_budget = self._get_budget(budget, deepcave_run)

        _objective = deepcave_run.get_objective(objective)
        assert _objective is not None

        cache_dir = DEEPCAVE_CACHE_DIR / approach_key / dataset
        if (cache_dir / f"footprint_{objective}_{budget}_configs_x.npy").exists():
            configs_x = np.load(
                cache_dir / f"footprint_{objective}_{budget}_configs_x.npy"
            )
            configs_y = np.load(
                cache_dir / f"footprint_{objective}_{budget}_configs_y.npy"
            )
            config_ids = np.load(
                cache_dir / f"footprint_{objective}_{budget}_config_ids.npy"
            )

            surface_x = np.load(
                cache_dir / f"footprint_{objective}_{budget}_surface_x.npy"
            )
            surface_y = np.load(
                cache_dir / f"footprint_{objective}_{budget}_surface_y.npy"
            )
            surface_z = np.load(
                cache_dir / f"footprint_{objective}_{budget}_surface_z.npy"
            )
        else:
            fp.calculate(
                objective=_objective,
                budget=selected_budget,
            )

            configs_x, configs_y, config_ids = fp.get_points()
            surface_x, surface_y, surface_z = fp.get_surface()

            cache_dir.mkdir(parents=True, exist_ok=True)

            np.save(
                cache_dir / f"footprint_{objective}_{budget}_configs_x.npy",
                configs_x
            )
            np.save(
                cache_dir / f"footprint_{objective}_{budget}_configs_y.npy",
                configs_y
            )
            np.save(
                cache_dir / f"footprint_{objective}_{budget}_config_ids.npy",
                config_ids
            )
            np.save(
                cache_dir / f"footprint_{objective}_{budget}_surface_x.npy",
                surface_x
            )
            np.save(
                cache_dir / f"footprint_{objective}_{budget}_surface_y.npy",
                surface_y
            )
            np.save(
                cache_dir / f"footprint_{objective}_{budget}_surface_z.npy",
                surface_z
            )

        z_min, z_max = np.min(surface_z), np.max(surface_z)

        z_min = np.floor(z_min / 10) * 10
        z_max = np.ceil(z_max / 10) * 10

        tick_step = 10
        ticks = np.arange(z_min, z_max + tick_step, tick_step)

        fig, ax = plt.subplots(1, 1, figsize=(self.figwidth / 2, self.figwidth / 2))
        cmap = sns.dark_palette(self.color_palette[0], reverse=True, as_cmap=True)
        heatmap = plt.contourf(surface_x, surface_y, surface_z, levels=100, cmap=cmap)
        cbar = plt.colorbar(heatmap, ticks=ticks)
        cbar.set_label(objective, labelpad=10)

        origins = ["Default", "Incumbent", *list(ORIGIN_MAP.values())]
        inc_config_id = incumbent["Configuration ID"].to_numpy()[-1]

        history_unique = history.drop_duplicates(subset="Configuration ID")

        config_origin = dict(
            zip(
                history_unique["Configuration ID"],
                history_unique["Config Origin"],
                strict=False
            )
        )
        if inc_config_id != 0:
            config_origin[inc_config_id] = "Incumbent"

        config_marker = dict(zip(origins, ["v", "^", "x", "x", "x"], strict=False))

        # We remove light blue from the palette here
        color_palette = self.color_palette[:5] + [self.color_palette[8]]
        config_color = dict(
            zip(origins, color_palette[1:len(origins) + 1], strict=False)
        )

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

        for x, y, _id in zip(configs_x, configs_y, config_ids, strict=False):
            _id = int(_id)
            origin = config_origin[_id]
            ax.scatter(
                x=x,
                y=y,
                label=config_origin[_id],
                marker=config_marker[origin],
                color=config_color[origin],
                s=60 * config_size[_id],
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

        self._format_axis(ax=ax)

        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")
        plt.title(f"Configuration Footprint for\n"
                  f"{APPROACH_REPLACE_MAP[approach_key].replace(' (ours)', '')} on "
                  f"{format_dataset_name(dataset)}")

        fig.subplots_adjust(
            top=0.89,
            bottom=0.25,
            left=0.03,
            right=0.93,
        )

        plt.savefig(
            self.analysis_plots[approach_key] / f"footprint_{dataset}.png",
            dpi=self.dpi
        )
        plt.clf()
        plt.close()

    def plot_footprints(
            self,
            approach_keys: list[str] | None = None,
            budget: int = COMBINED_BUDGET
        ) -> None:
        """Plot the configuration footprints for all datasets and approaches.

        Parameters
        ----------
        approach_keys : list[str] | None
            The approach keys to use. Defaults to None. In that case, all approaches are
            used.

        budget : int
            The budget to use. Defaults to COMBINED_BUDGET.
        """
        if approach_keys is None:
            approach_keys = APPROACHES
        for approach_key in approach_keys:
            for dataset in self.datasets:
                self._plot_footprint(
                    dataset=dataset,
                    approach_key=approach_key,
                    budget=budget
                )

    def _get_corr_categories(
            self,
            correlations: dict
        ) -> dict:
        """Get the correlation categories for a given correlation matrix.

        Parameters
        ----------
        correlations : dict
            The correlation matrix.

        Returns:
        -------
        dict
            The correlation categories.
        """
        categories = {}

        for budget1, budgets in correlations.items():
            for budget2, correlation in budgets.items():
                if budget1 == budget2:
                    continue

                if correlation >= 0.7:      # noqa: PLR2004
                    category = "Very strong"
                elif correlation >= 0.4:    # noqa: PLR2004
                    category = "Strong"
                elif correlation >= 0.3:    # noqa: PLR2004
                    category = "Moderate"
                elif correlation >= 0.2:    # noqa: PLR2004
                    category = "Weak"
                else:
                    category = "Not given"

                budget1 = round(budget1)    # noqa: PLW2901
                budget2 = round(budget2)    # noqa: PLW2901

                key = (budget1, budget2)
                key2 = (budget2, budget1)
                if float(budget1) < float(budget2):
                    categories[key2] = category
                else:
                    categories[key] = category

        return categories

    def _plot_budget_correlations(      # noqa: PLR0915, C901
            self,
            dataset: str,
            approach_key: str,
            objective: str = O_DSC,
        ) -> None:
        """Plot the budget correlations for a given dataset and approach.

        Parameters
        ----------
        dataset : str
            The dataset to plot.

        approach_key : str
            The approach key (hpo, hpo_nas, hpo_hnas).

        objective : str
            The objective to use. Defaults to O_DSC.
        """
        def round_dict_keys(d: dict) -> dict:
            """Helper for rounding budget  keys."""
            new_dict = {}
            for k, v in d.items():
                with contextlib.suppress(ValueError):
                    k = round(float(k))         # noqa: PLW2901

                if isinstance(v, dict):
                    v = round_dict_keys(v)      # noqa: PLW2901

                new_dict[k] = v
            return new_dict

        try:
            deepcave_run, _, _ = self.get_deepcave_data(
                dataset=dataset,
                approach_key=approach_key,
                objective=objective
            )
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
        budgets = [round(float(b)) for b in budgets]
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
                label=f"Budget {budget}",
            )

        for i, b1 in enumerate(budgets):
            budget_x = []
            budget_y = []
            for _j, b2 in enumerate(budgets):
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
        ax.legend(
            handles=budget_legend_elements,
            title="Budget",
            loc="upper center",
            bbox_to_anchor=(0.75, -0.15),
            ncol=2,
            fancybox=False,
            shadow=False,
            frameon=False
        )

        self._format_axis(ax=ax, grid=True)

        # Customize the plot
        plt.xlabel("Budget")
        plt.ylabel("Correlation")
        plt.title(f"Budget Correlation for "\
                  f"{APPROACH_REPLACE_MAP[approach_key].replace(' (ours)', '')} "
                  f"on {format_dataset_name(dataset)}")

        fig.subplots_adjust(
            top=0.92,
            bottom=0.34,
            left=0.09,
            right=0.98,
        )

        plt.savefig(
            self.analysis_plots[approach_key] /\
                f"budget_correlation_{dataset}.{self.format}",
            format=self.format,
            dpi=self.dpi
        )
        plt.clf()

    def plot_budget_correlations(
            self,
            approach_keys: list[str] | None = None
        ) -> None:
        """Plot the budget correlations for all datasets and approaches.

        Parameters
        ----------
        approach_keys : list[str] | None
            The approach keys to use. Defaults to None. In that case, all approaches are
            used.
        """
        if approach_keys is None:
            approach_keys = APPROACHES
        for approach_key in approach_keys:
            for dataset in self.datasets:
                self._plot_budget_correlations(
                    dataset=dataset,
                    approach_key=approach_key
                )

    def _get_joint_dataset_features(            # noqa: PLR0915
            self,
            budget: int = COMBINED_BUDGET,
            recompute: bool = False             # noqa: FBT001, FBT002
        ) -> pd.DataFrame:
        """Returns the joint dataset features and hyperparameter importances for all
        datasets.

        Parameters
        ----------
        budget : int
            The budget to use. Defaults to COMBINED_BUDGET.

        recompute : bool
            Whether to recompute the joint dataset features. Defaults to False.

        Returns:
        -------
        pd.DataFrame
            The resulting joint dataset features.
        """
        path = DEEPCAVE_CACHE_DIR / "dataset_features" / "joint_dataset_features.csv"
        if not recompute and path.exists():
            return pd.read_csv(path)
        (DEEPCAVE_CACHE_DIR / "dataset_features").mkdir(parents=True, exist_ok=True)

        all_baseline_metrics = []
        all_dataset_features = []
        all_hp_incumbents = []
        all_hp_importances = []

        for dataset in tqdm(self.datasets, desc="Processing dataset features"):
            baseline_metrics = self.get_baseline_data(dataset=dataset).metrics

            # We ignore MedSAM2 here
            baseline_metrics = baseline_metrics[
                baseline_metrics["Approach"] != "MedSAM2"
            ]

            baseline_metrics = baseline_metrics.drop(
                columns=["Approach", "Fold", "Dataset", "Mean"]
            )

            # We need to extract class labels and mean DSC
            baseline_metrics = baseline_metrics.mean().reset_index()
            baseline_metrics.columns = ["Class Label", "nnU-Net Val. DSC"]
            baseline_metrics["Dataset"] = dataset

            all_baseline_metrics += [baseline_metrics]

            dataset_features = extract_dataset_features(dataset=dataset)
            dataset_features = dataset_features[dataset_features["class_idx"] != 0]
            dataset_features["Dataset"] = dataset
            dataset_features["n_images"] = dataset_features["n_training_samples"] \
                + dataset_features["n_test_samples"]
            all_dataset_features += [dataset_features]

            importances_data = []
            incumbent_data = []

            for data_func in [
                self.get_hpo_data,
                self.get_nas_data,
                self.get_hnas_data
            ]:
                deepcave_run = data_func(dataset).deepcave_runs[dataset]

                # 1) We extract incumbent values for each HP
                objective = deepcave_run.get_objective(0)
                incumbent_cfg = deepcave_run.get_incumbent(objectives=objective)[0]

                for hp in incumbent_cfg:
                    assert isinstance(hp, str)

                    if isinstance(
                            deepcave_run.configspace[hp],
                            CategoricalHyperparameter
                        ):
                        continue

                    incumbent_data += [{
                        "Dataset": dataset,
                        "Hyperparameter": HYPERPARAMETER_REPLACEMENT_MAP[hp] \
                            + " Incumbent",
                        "Incumbent Value": incumbent_cfg[hp],
                    }]

                incumbents_df = pd.DataFrame(incumbent_data)
                all_hp_incumbents += [incumbents_df]

                # 2) We compute importance values for each HP
                selected_budget = self._get_budget(budget, deepcave_run)
                evaluator = fANOVA(run=deepcave_run)
                evaluator.calculate(budget=selected_budget, seed=42)

                importances = evaluator.get_importances(
                    hp_names=list(deepcave_run.configspace.keys())
                )

                for hp in importances:
                    assert isinstance(hp, str)

                    importance = importances[hp]
                    importances_data += [{
                        "Dataset": dataset,
                        "Hyperparameter": HYPERPARAMETER_REPLACEMENT_MAP[hp] \
                            + " Importance",
                        "Importance": importance[0],
                    }]

                importances_df = pd.DataFrame(importances_data)
                all_hp_importances += [importances_df]

        all_baseline_metrics = pd.concat(all_baseline_metrics)
        all_dataset_features = pd.concat(all_dataset_features)

        all_dataset_features = all_dataset_features[
            DATASET_FEATURES_REPLACEMENT_MAP.keys()
        ]
        all_dataset_features = all_dataset_features.rename(
            columns=DATASET_FEATURES_REPLACEMENT_MAP
        )

        numerical_cols = all_dataset_features.select_dtypes(include=["number"]).columns
        numerical_cols = numerical_cols.drop("Class Index")
        non_numerical_cols = all_dataset_features.select_dtypes(
            exclude=["number"]
        ).columns.difference(["Category"])
        non_numerical_cols = non_numerical_cols.drop("Dataset")
        non_numerical_cols = non_numerical_cols.drop("Class Label")
        non_numerical_cols = non_numerical_cols.drop("Instance")

        # Group by "category" and apply aggregation functions
        all_dataset_features = all_dataset_features.groupby(
            ["Dataset", "Class Label"]
        ).agg({
            **{col: "mean" for col in numerical_cols},
            **{col: "first" for col in non_numerical_cols}
        }).reset_index()

        joint_data = all_baseline_metrics.merge(
            all_dataset_features,
            on=["Dataset", "Class Label"],
            how="inner"
        )

        all_hp_incumbents = pd.concat(all_hp_incumbents)
        all_hp_incumbents = all_hp_incumbents.groupby(
            ["Dataset", "Hyperparameter"]
        ).mean().reset_index()
        all_hp_incumbents = all_hp_incumbents.pivot_table(
            columns="Hyperparameter",
            values="Incumbent Value",
            index="Dataset"
        ).reset_index()
        all_hp_incumbents.columns.name = None
        joint_data = joint_data.merge(
            all_hp_incumbents,
            on=["Dataset"],
            how="inner"
        )

        all_hp_importances = pd.concat(all_hp_importances)
        all_hp_importances = all_hp_importances.groupby(
            ["Dataset", "Hyperparameter"]
        ).mean().reset_index()
        all_hp_importances = all_hp_importances.pivot_table(
            columns="Hyperparameter",
            values="Importance",
            index="Dataset"
        ).reset_index()
        all_hp_importances.columns.name = None
        joint_data = joint_data.merge(
            all_hp_importances,
            on=["Dataset"],
            how="inner"
        )

        joint_data.to_csv(path, index=False)

        return joint_data

    @staticmethod
    def _filter_joint_dataset_features(
            features: pd.DataFrame,
            include: Literal["none", "incumbents", "importances"]
    ) -> pd.DataFrame:
        """Filters the joint dataset features based on the include parameter.

        Parameters
        ----------
        features : pd.DataFrame
            The joint dataset features.

        include : Literal["none", "incumbents", "importances"]
            The kind of hyperparameter values to include.
        """
        if include == "incumbents":
            # We filter all importances
            columns = [c for c in features.columns if "Importance" not in c]
            features = features[columns]
            features.columns = [c.replace(" Incumbent", "") for c in features.columns]
        elif include == "importances":
            # We filter all incumbents
            columns = [c for c in features.columns if "Incumbent" not in c]
            features = features[columns]
            features.columns = [c.replace(" Importance", "") for c in features.columns]
        elif include == "none":
            # We filter all incumbents and importances
            columns = [c for c in features.columns if "Incumbent" not in c \
                       and "Importance" not in c]
            features = features[columns]
        else:
            raise ValueError(f"Unknown value for include: {include}."
                             f"Must be one of 'none', 'incumbents' or 'importances'.")

        return features

    def plot_joint_dataset_features_heatmap(
            self,
            include: Literal["none", "incumbents", "importances"],
            orientation: Literal["left", "right"] = "right",
            corr_threshold: float = 0.7
    ) -> None:
        """Plot the joint dataset features heatmap.

        Parameters
        ----------
        include : Literal["none", "incumbents", "importances"]
            The kind of hyperparameter values to include.

        orientation : Literal["left", "right"]
            The orientation of the x-axis labels. Defaults to "right".

        corr_threshold : float
            The correlation threshold. Defaults to 0.7.
        """
        joint_data = self._get_joint_dataset_features()
        joint_data = self._filter_joint_dataset_features(
            features=joint_data,
            include=include
        )

        numerical_cols = joint_data.select_dtypes(include=["number"]).columns
        joint_data = joint_data[numerical_cols]

        if include == "none":
            figsize = (self.figwidth, self.figwidth * 0.5)
            title = "Correlation Heatmap for Baseline Metrics and "\
                "Dataset Properties"
        elif include == "incumbents":
            figsize = (self.figwidth, self.figwidth * 0.75)
            title = "Correlation Heatmap for  Baseline Metrics, "\
                "Dataset Properties and Incumbent Hyperparameter Values"
        else:
            figsize = (self.figwidth * 2, self.figwidth * 1.25)
            title = "Correlation Heatmap for Baseline Metrics, "\
                "Dataset Properties and Hyperparameter Importances"

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        corr = joint_data.corr(method="spearman")

        filtered_corr = corr.mask(corr == 1)

        valid_rows = (filtered_corr.abs() >= corr_threshold).any(axis=1)
        valid_cols = (filtered_corr.abs() >= corr_threshold).any(axis=0)

        filtered_corr = filtered_corr.loc[valid_rows, valid_cols]

        sns.heatmap(
            filtered_corr,
            cmap="coolwarm",
            annot=True,
            ax=ax,
            cbar=False,
            fmt=".2f",
        )

        if orientation == "left":
            plt.xticks(rotation=-90)
        else:
            plt.grid(visible=False)

        plt.title(title)
        plt.tight_layout()

        if include == "none":
            output_dir = self.dataset_analysis_plots
        else:
            output_dir = self.dataset_analysis_plots / include
            output_dir.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            output_dir / f"heatmap_{orientation}.{self.format}",
            format=self.format,
            dpi=self.dpi
        )

    def create_top_dataset_features_hps_table(
            self,
            include: Literal["incumbents", "importances"],
            corr_threshold: float = 0.7,
            plot_relationships: bool = False        # noqa: FBT001, FBT002
    ) -> None:
        """Create a table containing all correlations between a dataset feature
        and hyperparameter importance with an absolute value larger than the
        correlation threshold.

        Parameters
        ----------
        include : Literal["incumbents", "importances"]
            The kind of hyperparameter values to include.

        corr_threshold : float
            The correlation threshold. Defaults to 0.7.
        """
        joint_data = self._get_joint_dataset_features()
        joint_data = self._filter_joint_dataset_features(
            features=joint_data,
            include=include
        )
        numerical_cols = joint_data.select_dtypes(include=["number"]).columns
        joint_data = joint_data[numerical_cols]

        corr = joint_data.corr(method="spearman")

        # Extract relevant correlations
        correlations = []
        for hyperparameter in HYPERPARAMETER_REPLACEMENT_MAP.values():
            for dataset_feature in DATASET_FEATURES_REPLACEMENT_MAP.values():
                if not (dataset_feature in corr.index \
                        and hyperparameter in corr.columns):
                    continue

                corr_value = corr.loc[dataset_feature, hyperparameter]
                if np.abs(corr_value) >= corr_threshold:    # type: ignore
                    correlations.append({
                        "Hyperparameter": hyperparameter,
                        "Dataset Feature": dataset_feature,
                        "Correlation": corr_value
                })

        correlations = pd.DataFrame(correlations)
        correlations.to_latex(
            AUTONNUNET_TABLES / f"top_{include}_correlations.tex",
            float_format="%.2f"
        )

        if plot_relationships:
            for _, row in correlations.iterrows():
                self.plot_joint_dataset_features(
                    dataset_feature=row["Dataset Feature"],
                    hp_name=row["Hyperparameter"],
                    include=include
                )

    def plot_joint_dataset_features(
            self,
            dataset_feature: str,
            hp_name: str,
            include: Literal["incumbents", "importances"],
    ) -> None:
        """Plot the relationship between two dataset features, e.g. DSC and Class
        Volume Ratio.

        Parameters
        ----------
        feature_x : str
            The feature on the x-axis.

        feature_y : str
            The feature on the y-axis.
        """
        joint_data = self._get_joint_dataset_features()
        joint_data = self._filter_joint_dataset_features(
            features=joint_data,
            include=include
        )
        joint_data = joint_data[[dataset_feature, hp_name, "Dataset"]]
        joint_data = joint_data.groupby("Dataset").mean().reset_index()
        joint_data["Dataset"] = joint_data["Dataset"].apply(
            lambda d: format_dataset_name(d)
        )

        fig, ax = plt.subplots(1, 1, figsize=(self.figwidth / 2, self.figwidth / 2))

        sns.scatterplot(
            x=dataset_feature,
            y=hp_name,
            data=joint_data,
            hue="Dataset",
            ax=ax,
            palette=self.color_palette,
        )

        if include == "incumbents" and hp_name in [
            "Initial LR",
            "Momentum (SGD)",
            "Weight Decay"
        ]:
            ax.set_yscale("log")
        else:
            pass

        self._format_axis(ax=ax, grid=True)
        if include == "incumbents":
            title = f"Relationship between {dataset_feature}\nand Incumbent {hp_name}"
        else:
            title = f"Relationship between {dataset_feature}\nand {hp_name} Importance"

        plt.title(title)

        fig.subplots_adjust(
            top=0.89,
            bottom=0.43,
            left=0.17,
            right=0.87,
        )

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=2,
            title="Dataset",
            fancybox=False,
            shadow=False,
            frameon=False
        )

        output_dir = self.dataset_analysis_plots / include
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_dir /\
                f"{dataset_feature}_{hp_name}_{include}.{self.format}",
            dpi=self.dpi,
            format=self.format
        )
        plt.clf()
        plt.close()

    def create_emissions_table(
            self,
            key: str = "emissions"
        ) -> None:
        """Create the emissions table for the given key.

        Parameters
        ----------
        key : str
            The CodeCarbon emissions.csv column to use. Defaults to "emissions".
        """
        baseline_emissions = self._baseline_data.emissions[
            ["run_id", "Approach", "Fold", "Dataset", key]
        ]
        hpo_emissions = self._hpo_data.emissions[
            ["run_id", "Approach", "Fold", "Dataset", key]
        ]
        nas_emissions = self._nas_data.emissions[
            ["run_id", "Approach", "Fold", "Dataset", key]
        ]
        hnas_emissions = self._hnas_data.emissions[
            ["run_id", "Approach", "Fold", "Dataset", key]
        ]

        emissions = pd.concat(
            [baseline_emissions, hpo_emissions, nas_emissions, hnas_emissions]
        )
        emissions["Dataset"] = emissions["Dataset"].apply(
            lambda d: format_dataset_name(d)[:3]
        )

        # We want to go from seconds to hours
        if key == "duration":
            emissions.loc[:, key] = emissions[key] / 3600

        emissions = emissions.drop(columns=["Fold", "run_id"])

        emissions_per_dataset = emissions.groupby(
            ["Dataset", "Approach"]
        )[key].sum().reset_index()

        overall_sum = emissions.groupby("Approach")[key].sum().reset_index()
        overall_sum["Dataset"] = "Sum"

        emissions_combined = pd.concat(
            [emissions_per_dataset, overall_sum],
            ignore_index=True
        )
        emissions_table = emissions_combined.pivot_table(
            index="Dataset",
            columns="Approach",
            values=key
        ).fillna(0)

        approaches = np.array(list(APPROACH_REPLACE_MAP.values()))
        emissions_table = emissions_table.loc[:, approaches]

        emissions_table.loc[:, "\\textbf{Sum}"] = emissions_table.sum(axis=1)

        def format_number(n):
            n = round(n, 2)
            formatted = f"{n:,}".replace(",", r"\;")
            return f"${formatted}$"

        emissions_table = emissions_table.map(format_number)

        emissions_table.to_latex(
            AUTONNUNET_TABLES / f"total_{key}.tex",
            float_format="%.2f"
        )

    def _create_dataset_dsc_table(
            self,
            dataset: str
        ) -> None:
        """Create the validation DSC table for a given dataset.

        Parameters
        ----------
        dataset : str
            The dataset to create the table for.
        """
        def format_approach(a: str) -> str:
            return a.replace(" (ours)", "").replace("nnU-Net (", "").replace(")", "")

        baseline_metrics = self.get_baseline_data(dataset=dataset).metrics
        hpo_data = self.get_hpo_data(dataset=dataset).incumbent_metrics
        nas_data = self.get_nas_data(dataset=dataset).incumbent_metrics
        hnas_data = self.get_hnas_data(dataset=dataset).incumbent_metrics

        dataset_results = pd.concat([baseline_metrics, hpo_data, nas_data, hnas_data])

        # We compute the mean over all folds
        all_results = []
        for approach in (APPROACH_REPLACE_MAP.values()):
            _data = dataset_results[dataset_results["Approach"] == approach]
            _data = _data.drop(columns=["Approach", "Fold", "Dataset", "Mean"])

            # We need to drop all class labels that are not
            # present in the dataset
            _data = _data.dropna(axis=1, how="all")

            # Average over folds
            _data = _data.mean().to_frame().T

            _data["Approach"] = format_approach(approach)
            all_results.append(_data)

        all_results = pd.concat(all_results)
        class_labels = [c for c in all_results.columns if c not in ["Approach"]]

        for c in class_labels:
            all_results.loc[:, c] = (all_results[c]).astype(float)

        melted_results = all_results.melt(
            id_vars=["Approach"],
            value_vars=class_labels,
            var_name="Class Label",
            value_name="Mean"
        )

        pivot_table = melted_results.pivot_table(
            index="Class Label",
            columns="Approach",
            values="Mean"
        )
        pivot_table = pivot_table[
            [format_approach(a) for a in APPROACH_REPLACE_MAP.values()]
        ]

        pivot_table.loc["\\textbf{Mean}", :] = pivot_table.mean(axis=0)

        def highlight_max(row):
            max_val = row.max()
            return pd.Series(
                [
                    f"$\\mathbf{{{val:.2f}}}$" \
                        if val == max_val else f"${val:.2f}$" for val in row
                ],
                index=row.index
            )

        pivot_table = pivot_table.apply(highlight_max, axis=1)

        pivot_table.to_latex(
            AUTONNUNET_TABLES / f"results_{dataset}.tex",
            float_format="%.2f",
            caption="Performance Comparison",
            label="tab:results"
        )

    def create_dataset_dsc_tables(self):
        """Create the validation DSC tables for all datasets."""
        for dataset in self.datasets:
            self._create_dataset_dsc_table(dataset)

    def create_dsc_table(self):
        """Create the overall validation DSC table."""
        def format_approach(a: str) -> str:
            return a.replace(" (ours)", "").replace("nnU-Net (", "").replace(")", "")

        all_results = []
        for dataset in self.datasets:
            baseline_metrics = self.get_baseline_data(dataset=dataset).metrics
            hpo_data = self.get_hpo_data(dataset=dataset).incumbent_metrics
            nas_data = self.get_nas_data(dataset=dataset).incumbent_metrics
            hnas_data = self.get_hnas_data(dataset=dataset).incumbent_metrics

            dataset_results = pd.concat(
                [baseline_metrics, hpo_data, nas_data, hnas_data]
            )

            # We compute the mean over all folds
            for approach in (APPROACH_REPLACE_MAP.values()):
                _data = dataset_results[dataset_results["Approach"] == approach]
                _data = _data.drop(columns=["Approach", "Fold", "Dataset"])
                _data = _data[["Mean"]]
                _data = _data.mean().to_frame().T
                _data["Approach"] = format_approach(approach)
                _data["Dataset"] = format_dataset_name(dataset)[:3]
                all_results.append(_data)

        all_results = pd.concat(all_results)
        all_results.loc[:, "Mean"] = (all_results["Mean"]).astype(float)

        pivot_table = all_results.pivot_table(
            index="Dataset",
            columns="Approach",
            values="Mean"
        )
        pivot_table = pivot_table[
            [format_approach(a) for a in APPROACH_REPLACE_MAP.values()]
        ]

        pivot_table.loc[ "\\textbf{Mean}", :] = pivot_table.mean(axis=0)

        def highlight_max(row):
            max_val = row.max()
            return pd.Series(
                [
                    f"$\\mathbf{{{val:.2f}}}$" \
                        if val == max_val else f"${val:.2f}$" for val in row
                ],
                index=row.index
            )

        pivot_table = pivot_table.apply(highlight_max, axis=1)

        pivot_table.to_latex(
            AUTONNUNET_TABLES / "results_dsc.tex",
            float_format="%.2f",
            caption="Performance Comparison",
            label="tab:results"
        )

    def create_runtime_table(self):
        """Create the overall runtime table."""
        def format_approach(a: str) -> str:
            return a.replace(" (ours)", "").replace("nnU-Net (", "").replace(")", "")

        all_results = []
        for dataset in self.datasets:
            baseline_progress = self.get_baseline_data(dataset=dataset).progress
            hpo_data = self.get_hpo_data(dataset=dataset).incumbent_progress
            nas_data = self.get_nas_data(dataset=dataset).incumbent_progress
            hnas_data = self.get_hnas_data(dataset=dataset).incumbent_progress
            dataset_progress = pd.concat(
                [baseline_progress, hpo_data, nas_data, hnas_data]
            )

            # We compute the mean over all folds
            for approach in (APPROACH_REPLACE_MAP.values()):
                # We skip MedSAM2 here since runtimes are not directly comparable
                if approach == "MedSAM2":
                    continue

                _data = dataset_progress[dataset_progress["Approach"] == approach]

                _data = _data[["Runtime"]]
                _data.loc[:, "Runtime"] = _data["Runtime"] / 3600 / self.n_folds
                _data = _data.sum().to_frame().T
                _data["Approach"] = format_approach(approach)
                _data["Dataset"] = format_dataset_name(dataset)[:3]

                all_results.append(_data)

        all_results = pd.concat(all_results)
        all_results.loc[:, "Runtime"] = (all_results["Runtime"]).astype(float)

        pivot_table = all_results.pivot_table(
            index="Dataset",
            columns="Approach",
            values="Runtime"
        )
        pivot_table = pivot_table[
            [format_approach(a) for a in APPROACH_REPLACE_MAP.values() \
                if format_approach(a) != "MedSAM2"]
        ]

        pivot_table.loc[ "\\textbf{Mean}", :] = pivot_table.mean(axis=0)
        pivot_table["Speedup"] = np.inf

        def highlight_min(row):
            min_val = row[:-1].min()
            speedup = row["Conv"] / min_val
            return pd.Series(
                [
                    f"$\\mathbf{{{val:.2f}}}$" \
                        if val == min_val else f"${val:.2f}$" for val in row[:-1]
                ] + [speedup],
                index=row.index
            )

        pivot_table = pivot_table.apply(highlight_min, axis=1)

        pivot_table.to_latex(
            AUTONNUNET_TABLES / "results_runtime.tex",
            float_format="%.2f",
            caption="Performance Comparison",
            label="tab:results"
        )

    def _read_msd_results(
            self,
            approaches: list[str] | None = None
        ) -> pd.DataFrame:
        """Read the MSD test set results for the given approaches.

        Parameters
        ----------
        approaches : list[str]
            The approaches to read the results for. Defaults to None. In that case, all
            approaches are used.

        Returns:
        -------
        pd.DataFrame
            The resulting MSD results.
        """
        if approaches is None:
            approaches = APPROACHES

        results = []

        for approach in approaches:
            result_path = AUTONNUNET_MSD_RESULTS /\
                f"{approach}_{self.configuration}.json"
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
        """Compare the MSD test set results for all approaches."""
        approaches = ["baseline_ConvolutionalEncoder", "hpo", "hpo_nas"]
        msd_results = self._read_msd_results(approaches)


        msd_results_per_dataset = msd_results[["Approach", "Dataset", "mean"]].groupby(
            ["Approach", "Dataset"]
        ).mean().reset_index()
        msd_results_per_dataset = msd_results_per_dataset.rename(
            columns={"mean": "DSC"}
        )

        print(msd_results_per_dataset)

    def plot_msd_overview(self) -> None:    # noqa: PLR0915
        """Plot an overview of the MSD datasets."""
        def get_slice(img, label):
            img_data = img.get_fdata()
            label_data = label.get_fdata()

            _label_data = label_data.copy()
            _label_data[label_data != 0] = 1

            if len (img_data.shape) == 4:   # noqa: PLR2004
                img_data = img_data[:, :, :, 0]

            max_foreground = 0
            slice_idx = 0
            for i in range(_label_data.shape[2]):
                if (fg_sum := _label_data[:, :, i].sum()) > max_foreground:
                    max_foreground = fg_sum
                    slice_idx = i

            img_slice = img_data[:, :, slice_idx]
            label_slice = label_data[:, :, slice_idx]

            return img_slice, label_slice

        def get_slices(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            dataset_path = NNUNET_DATASETS / dataset_name_to_msd_task(dataset)

            for file in (dataset_path / "imagesTr" ).glob("*.nii.gz"):
                if file.name.startswith("._"):
                    continue

                img = nib.loadsave.load(file)
                label = nib.loadsave.load(dataset_path / "labelsTr" / file.name)
                spacing = np.abs(np.diag(img.affine)[:3])   # type: ignore

                img_slice, label_slice = get_slice(img, label)
                label_slice[label_slice == 0] = np.nan

                # We just use the first image
                break

            return img_slice, label_slice, spacing

        fig, axs = plt.subplots(
            ncols=5,
            nrows=2,
            figsize=(8, 4.25),
        )

        output_dir = AUTONNUNET_PLOTS / "related_work"
        output_dir.mkdir(parents=True, exist_ok=True)

        images = []
        labels = []

        viridis = plt.get_cmap("viridis")

        for dataset, ax in zip(self.datasets, axs.flatten(), strict=False):
            img, label, spacing = get_slices(dataset)

            colors = list(viridis(np.linspace(1, 0.2, len(np.unique(label)))))
            custom_cmap = mcolors.ListedColormap(colors)

            images.append(img)
            labels.append(label)

            ax.set_title(format_dataset_name(dataset).replace(" ", "\n"))

            aspect = spacing[1] / spacing[0]
            ax.imshow(img, cmap="gray", aspect=aspect)
            ax.imshow(label, cmap=custom_cmap, alpha=0.7, aspect=aspect)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(
            output_dir / f"msd_overview.{self.format}",
            dpi=self.dpi,
            format=self.format
        )
        plt.clf()

        for dataset, img, label in zip(self.datasets, images, labels, strict=False):
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))

            ax.imshow(img, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")

            plt.tight_layout()
            plt.savefig(
                output_dir / f"{dataset}_img.{self.format}",
                dpi=self.dpi,
                format=self.format
            )
            plt.clf()

            fig, ax = plt.subplots(1, 1, figsize=(4, 4))

            colors = list(viridis(np.linspace(1, 0.2, len(np.unique(label)))))
            custom_cmap = mcolors.ListedColormap(colors)

            ax.imshow(img, cmap="gray")
            ax.imshow(label, cmap="viridis", alpha=0.7)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis("off")

            plt.tight_layout()
            plt.savefig(
                output_dir / f"{dataset}_label.{self.format}",
                dpi=self.dpi,
                format=self.format
            )
            plt.close()

    def plot_qualitative_segmentations(self) -> None:   # noqa: PLR0915, C901
        """Plot the qualitative segmentations for all datasets."""
        def get_slice_idxs(gt_label: np.ndarray):
            def _get_slice_idx(label: np.ndarray, dim: int) -> int:
                max_foreground = 0
                slice_idx = 0

                _label = label.copy()
                _label[_label != 0] = 1

                for i in range(_label.shape[dim]):
                    index = tuple(
                        slice(None) if j != dim else i for j in range(_label.ndim)
                    )
                    if (fg_sum := _label[index].sum()) > max_foreground:
                        max_foreground = fg_sum
                        slice_idx = i

                return slice_idx

            return tuple(_get_slice_idx(gt_label, dim) for dim in range(3))

        def get_gt_data(
                dataset: str
            ) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
            dataset_path = NNUNET_DATASETS / dataset_name_to_msd_task(dataset)

            for file in (dataset_path / "imagesTr" ).glob("*.nii.gz"):
                if file.name.startswith("._"):
                    continue

                img = nib.loadsave.load(file)
                label = nib.loadsave.load(dataset_path / "labelsTr" / file.name)

                img_data: np.ndarray = img.get_fdata()      # type: ignore
                label_data: np.ndarray = label.get_fdata()  # type: ignore

                affine = img.affine                         # type: ignore
                voxel_spacing = np.abs(np.diag(affine)[:3])

                # We skip the very small examples for the visualization
                if dataset == "Dataset003_Liver" and img_data.shape[2] < 500:           # noqa: PLR2004
                    continue
                if dataset == "Dataset005_Prostate" and img_data.shape[2] < 24:         # noqa: PLR2004
                    continue
                if dataset == "Dataset007_Pancreas" and img_data.shape[2] < 137:        # noqa: PLR2004
                    continue
                if dataset == "Dataset008_HepaticVessel" and img_data.shape[2] < 138:   # noqa: PLR2004
                    continue
                if dataset == "Dataset009_Spleen" and img_data.shape[2] < 168:          # noqa: PLR2004
                    continue
                if dataset == "Dataset010_Colon" and "colon_061" not in file.name:
                    continue

                # In case of mp-MRI, we use the first setting
                if len(img_data.shape) == 4:    # noqa: PLR2004
                    img_data = img_data[:, :, :, 0]

                return file.name, img_data, label_data, voxel_spacing

            raise ValueError(f"No data found for {dataset}")

        def get_slices(         # noqa: PLR0912, C901
                dataset: str
            ) -> tuple[
                dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
                np.ndarray
            ]:
            slices = {}

            file_name, img_data, label_data, spacing = get_gt_data(dataset)

            x, y, z = get_slice_idxs(label_data)
            label_data[label_data == 0] = np.nan

            slices["Image"] = [
                img_data[:, :, z],
                img_data[x, :, :],
                img_data[:, y, :]
            ]

            slices["Ground\nTruth"] = [
                label_data[:, :, z],
                label_data[x, :, :],
                label_data[:, y, :]
            ]

            for approach, path in zip(
                list(APPROACH_REPLACE_MAP.values())[:4], [
                    self.baseline_conv,
                    self.baseline_resenc_m,
                    self.baseline_resenc_l,
                    self.baseline_medsam2
                ], strict=False):
                # Since we don't know in which fold the prediction is in the validation
                # split, we need to iterate over all folds and take the first match
                for fold in range(self.n_folds):
                    if approach == "MedSAM2":
                        _file_name = file_name.split(".nii.gz")[0] + "_0000_1.nii.gz"
                        file_path = path / dataset / f"fold_{fold}" / "validation" /\
                            _file_name

                    else:
                        file_path = path / dataset / self.configuration /\
                            f"fold_{fold}" / "validation" / file_name
                    if file_path.exists():
                        break

                if not file_path.exists():
                    continue

                if approach == "MedSAM2":
                    paths = list(file_path.parent.glob(file_path.stem[:-6] + "*"))

                    pred_data = np.full_like(
                        nib.loadsave.load(paths[0]).get_fdata(), np.nan)   # type: ignore
                    for i, _path in enumerate(paths):
                        pred = nib.loadsave.load(_path).get_fdata()      # type: ignore
                        pred_data[pred == 1] = i + 1
                else:
                    pred = nib.loadsave.load(file_path)
                    pred_data = pred.get_fdata()        # type: ignore

                pred_data[pred_data == 0] = np.nan

                _approach = approach.replace(" ", "\n")

                slices[_approach] = [
                    pred_data[:, :, z],
                    pred_data[x, :, :],
                    pred_data[:, y, :]
                ]

            for approach, path in zip(
                list(APPROACH_REPLACE_MAP.values())[4:], [
                    self.hpo_dir,
                    self.nas_dir,
                    self.hnas_dir
                ], strict=False):

                # Since we don't know in which fold the prediction is in the validation
                # split, we need to iterate over all folds and take the first match
                for fold in range(self.n_folds):
                    file_path = path / dataset / self.configuration / "0" /\
                        "incumbent" / f"fold_{fold}" / "validation" / file_name
                    if file_path.exists():
                        break

                if not file_path.exists():
                    continue

                pred = nib.loadsave.load(file_path)
                pred_data = pred.get_fdata()        # type: ignore
                pred_data[pred_data == 0] = np.nan

                _approach = approach.replace(" ", "\n").replace("+", "+\n")

                slices[_approach] = [
                    pred_data[:, :, z],
                    pred_data[x, :, :],
                    pred_data[:, y, :]
                ]

            return slices, spacing

        output_dir = AUTONNUNET_PLOTS / "qualitative_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        viridis = plt.get_cmap("viridis")

        for dataset in self.datasets:
            dataset_info_path = self.baseline_conv / dataset / self.configuration /\
                "fold_0" / "dataset.json"
            dataset_info = load_json(dataset_info_path)
            labels = {v - 1: k for k, v in dataset_info["labels"].items() if v > 0}

            # Uppercase labels first characters
            labels = {k: v[0].upper() + v[1:] for k, v in labels.items()}

            colors = list(viridis(np.linspace(1, 0.3, len(labels))))
            custom_cmap = mcolors.ListedColormap(colors)

            patches = [
                mpatches.Patch(
                    color=colors[label],
                    label=labels[label]
                ) for label in labels
            ]

            slices, spacing = get_slices(dataset)
            ndim = len(slices["Image"])

            aspect_ratios = {
                0: spacing[1] / spacing[0],
                1: spacing[2] / spacing[0],
                2: spacing[2] / spacing[1]
            }

            fig, axs = plt.subplots(
                ncols=len(slices),
                nrows=ndim,
                figsize=(self.figwidth, 4.25),
            )

            for i, slice_name in enumerate(slices):
                for j, slice_data in enumerate(slices[slice_name]):
                    ax = axs[j, i]
                    aspect_ratio = aspect_ratios[j]

                    if j > 0:
                        _slice_data = np.rot90(slice_data)
                        img = np.rot90(slices["Image"][j])
                    else:
                        _slice_data = slice_data
                        img = slices["Image"][j]

                    ax.imshow(_slice_data, cmap="gray", aspect=aspect_ratio)

                    if slice_name != "Image":
                        ax.imshow(img, cmap="gray", aspect=aspect_ratio)
                        ax.imshow(_slice_data, cmap=custom_cmap, aspect=aspect_ratio)

                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.axis("off")

                    if j == 0:
                        ax.set_title(slice_name)

            fig.suptitle(f"Qualitative Results for {format_dataset_name(dataset)}")
            fig.legend(
                handles=patches,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.07),
                ncol=len(labels),
                fancybox=False,
                shadow=False,
                frameon=False
            )
            fig.subplots_adjust(
                top=0.8,
                bottom=0.08,
                left=0.,
                right=1.,
                hspace=0.07,
                wspace=0.1,
            )

            plt.savefig(output_dir /\
                        f"{dataset}.{self.format}", dpi=self.dpi, format=self.format)
            plt.clf()
            plt.close()

    def plot_dsc_exmaple(self):
        """Plot an example for the DSC calculation."""
        legend_labels = {
            0: "Background",
            1: "X (Ground Truth)",
            2: "Y (Prediction)",
            3: "X ∩ Y (Intersection)"
        }

        viridis = plt.get_cmap("viridis")

        colors = [
            (234 / 255, 234 / 255, 241 / 255, 1),
            *list(viridis(np.linspace(1, 0.2, 3)))
        ]

        custom_cmap = mcolors.ListedColormap(colors)

        patches = [
            mpatches.Patch(
                color=colors[val],
                label=legend_labels[val]
            ) for val in legend_labels
        ]

        def create_circle_image(size, radius, center):
            x, y = np.meshgrid(np.arange(size), np.arange(size))
            return ((x - center[0])**2 + (y - center[1])**2) <= radius**2

        def dsc(gt, pred):
            intersection = np.logical_and(gt, pred).sum()
            return (2. * intersection) / (gt.sum() + pred.sum())

        size = 128
        radius = 0.25 * size
        center = (size // 2, size // 2)

        large_gt = create_circle_image(size, radius=radius, center=center)
        small_gt = create_circle_image(size, radius=radius, center=center)

        large_pred_good = create_circle_image(
            size,
            radius=radius * 1.1,
            center=center
        )
        large_pred_bad = create_circle_image(
            size,
            radius=radius * 1.5,
            center=(size // 2 + radius // 3, size // 2 + radius // 3)
        )
        small_pred_good = create_circle_image(
            size,
            radius=radius * 0.9,
            center=(size // 2 + radius // 2, size // 2 + radius // 2)
        )
        small_pred_bad = create_circle_image(
            size,
            radius=radius * 0.5,
            center=(size // 2 + radius // 3, size // 2 + radius // 3)
        )

        dsc_scores = [
            dsc(large_gt, large_pred_good),
            dsc(large_gt, large_pred_bad),
            dsc(small_gt, small_pred_good),
            dsc(small_gt, small_pred_bad)
        ]

        titles = [
            f"DSC(X,Y) = {dsc_scores[0]:.2f}",
            f"DSC(X,Y) = {dsc_scores[1]:.2f}",
            f"DSC(X,Y) = {dsc_scores[2]:.2f}",
            f"DSC(X,Y) = {dsc_scores[3]:.2f}"
        ]

        fig, axes = plt.subplots(1, 4, figsize=(8, 2.5))

        for i, (ax, pred, title) in enumerate(
            zip(
                axes,
                [large_pred_good, large_pred_bad, small_pred_good, small_pred_bad],
                titles,
                strict=False
            )
        ):
            gt = large_gt if i < 2 else small_gt    # noqa: PLR2004

            combined = gt.astype(int) + pred.astype(int) * 2

            ax.imshow(combined, cmap=custom_cmap)

            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.legend(
            handles=patches,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.12),
            ncol=4,
            fancybox=False,
            shadow=False,
            frameon=False
        )
        output_dir = AUTONNUNET_PLOTS / "related_work"
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(output_dir / f"dsc.{self.format}", dpi=self.dpi, format=self.format)

    def _load_cross_eval_matrix(self) -> pd.DataFrame:
        """Loads the matrix of evaluation returns for the cross-evaluation 
        of HPO+NAS incumbents and datasets.

        Returns
        -------
        pd.DataFrame
            The cross-evaluation matrix.
        """
        matrix = pd.DataFrame(
            columns=self.datasets,
            index=self.datasets
        )

        for dataset_cfg in self.datasets:
            for dataset_eval in self.datasets:
                # For the diagonal, we just load the original HPO+NAS result
                if dataset_cfg == dataset_eval:
                    incumbents = self.get_hnas_data(
                        dataset=dataset_cfg).incumbents[self.objectives[0]]
                    
                    # In the incumbents.csv, we store the the cost (1 - DSC [%]),
                    dsc = 100 - incumbents[O_DSC].iloc[-1]

                    matrix.loc[dataset_eval, dataset_cfg] = dsc
                    continue

                # For the remaining entries, we need to load the actual 
                # cross-evaluation results
                base_dir = self.cross_eval_dir / dataset_cfg / dataset_eval /\
                    self.configuration / str(self.hpo_seed) / "incumbent"

                all_metrics = []
                for fold in range(self.n_folds):
                    fold_dir = base_dir / f"fold_{fold}"
                    if not (fold_dir / "validation").exists():
                        continue

                    metrics = self._load_nnunet_metrics(fold_dir)
                    all_metrics += [metrics]

                if len(all_metrics) == 0:
                    dsc = np.nan
                else:
                    all_metrics = pd.concat(all_metrics)
                    dsc = metrics["Mean"].mean()

                matrix.loc[dataset_eval, dataset_cfg] = dsc

        return matrix
    
    def plot_cross_eval_matrix(self) -> None:
        """Plots the matrix of evaluation returns for the cross-evaluation 
        of HPO+NAS incumbents and datasets.
        """
        matrix = self._load_cross_eval_matrix()

        print(matrix)
        exit()

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        sns.heatmap(
            matrix.astype(float),
            annot=True,
            fmt=".2f",
            cmap="viridis",
            ax=ax
        )

        ax.set_title("Cross-Evaluation of HPO+NAS Incumbents")
        ax.set_xlabel("Evaluation Dataset")
        ax.set_ylabel("Configuration Dataset")

        output_dir = AUTONNUNET_PLOTS / "cross_eval"
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(output_dir / f"cross_eval_matrix.{self.format}", dpi=self.dpi, format=self.format)

    def _compute_hp_interactions(
            self,
            dataset: str,
            approach_key: str,
            budget: int = COMBINED_BUDGET
    ) -> pd.DataFrame:
        """Computes the interactions between hyperparameters for all datasets
        for a given approach.

        Parameters
        ----------
        dataset : str
            The dataset to use for the evaluation.

        approach_key : str
            The key of the approach to use.

        budget : int
            The budget to use for the evaluation. Defaults to COMBINED_BUDGET.

        Returns
        -------
        pd.DataFrame
            The resulting interactions.
        """
        if approach_key == "hpo":
            data_func = self.get_hpo_data
        elif approach_key == "hpo_nas":
            data_func = self.get_nas_data
        elif approach_key == "hpo_hnas":
            data_func = self.get_hnas_data
        else:
            raise ValueError(f"Unknown approach key {approach_key}")

        deepcave_run = data_func(
            dataset=dataset).deepcave_runs[dataset]
        
        selected_budget = self._get_budget(budget, deepcave_run)

        evaluator = fANOVA(run=deepcave_run)
        evaluator.calculate(budget=selected_budget, seed=42)


        hyperparameters = [
            HYPERPARAMETER_REPLACEMENT_MAP[hp] \
                for hp in deepcave_run.configspace.keys()
        ]
        interactions = pd.DataFrame(
            [],
            columns=hyperparameters,
            index=hyperparameters
        )

        fANOVA_interactions = evaluator.get_most_important_pairwise_marginals(n=-1)
        for (hp1, hp2), percentage in fANOVA_interactions.items():
            if percentage < 0.05:
                continue
            interactions.loc[hp1, hp2] = percentage * 100
            interactions.loc[hp2, hp1] = percentage * 100

        interactions = pd.DataFrame(interactions)

        return interactions
    
    def create_hp_interaction_tables(
            self,
            approach_key: str,
            budget: int = COMBINED_BUDGET
    ):
        """Creates the hyperparameter interaction tables for all datasets
        for a given approach.

        Parameters
        ----------
        approach_key : str
            The key of the approach to use.

        budget : int
            The budget to use for the evaluation. Defaults to COMBINED_BUDGET.
        """
        for dataset in self.datasets:
            interactions = self._compute_hp_interactions(
                dataset=dataset,
                approach_key=approach_key,
                budget=budget
            )
            print(interactions)
            exit()

            output_dir = AUTONNUNET_TABLES / approach_key
            output_dir.mkdir(parents=True, exist_ok=True)

            interactions.to_latex(
                output_dir / f"interactions_{dataset}.tex",
                float_format="%.2f"
            )