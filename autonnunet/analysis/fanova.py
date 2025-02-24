"""DeepCAVE fANOVA implementation."""
# This Code is based on https://github.com/automl/DeepCAVE/blob/main/deepcave/evaluators/fanova.py
from __future__ import annotations

import itertools as it
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

import numpy as np
from deepcave.constants import COMBINED_COST_NAME
from deepcave.evaluators.epm.fanova_forest import FanovaForest
from deepcave.utils.logs import get_logger

if TYPE_CHECKING:
    from deepcave.runs import AbstractRun
    from deepcave.runs.objective import Objective


class fANOVA:       # noqa: N801
    """Calculate and provide midpoints and sizes.

    They are generated from the forest's split values in order to get the marginals.

    Properties
    ----------
    run : AbstractRun
        The Abstract Run used for the calculation.
    cs : ConfigurationSpace
        The configuration space of the run.
    hps : List[Hyperparameters]
        The Hyperparameters of the configuration space.
    hp_names : List[str]
        The corresponding names of the Hyperparameters.
    n_trees : int
        The number of trees.
    """

    def __init__(self, run: AbstractRun):   # noqa: D107
        if run.configspace is None:
            raise RuntimeError("The run needs to be initialized.")

        self.run = run
        self.cs = run.configspace
        self.hps = list(self.cs.values())
        self.hp_names = list(self.cs.keys())
        self.logger = get_logger(self.__class__.__name__)
        self.n_dims = len(self.hps)

    def calculate(
        self,
        objectives: Objective | list[Objective] | None = None,
        budget: int | float | None = None,
        n_trees: int = 16,
        seed: int = 0,
    ) -> None:
        """Get the data with respect to budget and train the forest on the
        encoded data.

        Note:
        ----
        Right now, only `n_trees` is used. It can be further specified
        if needed.

        Parameters
        ----------
        objectives : Optional[Union[Objective, List[Objective]]], optional
            Considered objectives. By default None. If None, all objectives
            are considered.
        budget : Optional[Union[int, float]], optional
            Considered budget. By default None. If None, the highest budget
            is chosen.
        n_trees : int, optional
            How many trees should be used. By default 16.
        seed : int
            Random seed. By default 0.
        """
        if objectives is None:
            objectives = self.run.get_objectives()

        if budget is None:
            budget = self.run.get_highest_budget()

        self.n_trees = n_trees

        # Get data
        df = self.run.get_encoded_data(     # noqa: PD901
            objectives, budget, specific=True, include_combined_cost=True
        )
        X = df[self.hp_names].to_numpy()
        # Combined cost name includes the cost of all selected objectives
        Y = df[COMBINED_COST_NAME].to_numpy()

        # Get model and train it
        self._model = FanovaForest(self.cs, n_trees=n_trees, seed=seed)
        self._model.train(X, Y)

    def get_importances(
        self,
        hp_names: list[str] | None = None,
        depth: int = 1,
        sort: bool = True       # noqa: FBT001, FBT002
    ) -> dict[str | tuple[str, ...], tuple[float, float, float, float]]:
        """Return the importance scores from the passed Hyperparameter names.

        Warning:
        -------
        Using a depth higher than 1 might take much longer.

        Parameters
        ----------
        hp_names : Optional[List[str]]
            Selected Hyperparameter names to get the importance scores from.
            If None, all Hyperparameters of the configuration space are used.
        depth : int, optional
            How often dimensions should be combined. By default 1.
        sort : bool, optional
            Whether the Hyperparameters should be sorted by importance. By default True.

        Returns:
        -------
        Dict[Union[str, Tuple[str, ...]], Tuple[float, float, float, float]]
            Dictionary with Hyperparameter names and the corresponding
            importance scores. The values are tuples of the form (mean
            individual, var individual, mean total, var total).
            Note that individual and total are the same if depth is 1.

        Raises:
        ------
        RuntimeError
            If there is zero total variance in all trees.
        """
        if hp_names is None:
            hp_names = self.cs.get_hyperparameter_names()

        hp_ids = []
        for hp_name in hp_names:
            hp_ids.append(self.cs.index_of[hp_name])

        # Calculate the marginals
        vu_individual, vu_total = self._model.compute_marginals(hp_ids, depth)

        importances: dict[tuple[Any, ...], tuple[float, float, float, float]] = {}
        for k in range(1, len(hp_ids) + 1):
            if k > depth:
                break

            for sub_hp_ids in it.combinations(hp_ids, k):
                sub_hp_ids = tuple(sub_hp_ids)  # noqa: PLW2901

                # clean here to catch zero variance in a trees
                non_zero_idx = np.nonzero(
                    [self._model.trees_total_variance[t] for t in range(self.n_trees)]
                )

                if len(non_zero_idx[0]) == 0:
                    self.logger.warning("Encountered zero total variance in all trees.")
                    importances[sub_hp_ids] = (
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    )
                    continue

                fractions_total = np.array(
                    [
                        vu_total[sub_hp_ids][t] / self._model.trees_total_variance[t]
                        for t in non_zero_idx[0]
                    ]
                )
                fractions_individual = np.array(
                    [
                        vu_individual[sub_hp_ids][t] \
                            / self._model.trees_total_variance[t]
                        for t in non_zero_idx[0]
                    ]
                )

                importances[sub_hp_ids] = (
                    np.mean(fractions_individual),
                    np.var(fractions_individual),
                    np.mean(fractions_total),
                    np.var(fractions_total),
                )

        # Sort by total mean fraction
        if sort:
            importances = dict(sorted(importances.items(), key=lambda item: item[1][2]))

        # The ids get replaced with hyperparameter names again
        all_hp_names = list(self.cs.keys())
        importances_: dict[
            str | tuple[str, ...],
            tuple[float, float, float, float]
        ] = {}
        for hp_ids_importances, values in importances.items():
            hp_names = [all_hp_names[hp_id] for hp_id in hp_ids_importances]
            hp_names_key: tuple[str, ...] | str
            hp_names_key = hp_names[0] if len(hp_names) == 1 else tuple(hp_names)
            importances_[hp_names_key] = values

        return importances_

    def marginal_mean_variance_for_values(self, dimlist, values_to_predict):
        """Return the marginal of selected parameters for specific values.

        Parameters
        ----------
        dimlist: list
                Contains the indices of ConfigSpace for the selected parameters
                (starts with 0)
        values_to_predict: list
                Contains the values to be predicted

        Returns:
        -------
        tuple
            marginal mean prediction and corresponding variance estimate
        """
        sample = np.full(self.n_dims, np.nan, dtype=np.float32)
        for i in range(len(dimlist)):
            sample[dimlist[i]] = values_to_predict[i]

    def get_most_important_pairwise_marginals(self, n: int = -1):
        """Return the n most important pairwise marginals from the whole ConfigSpace.

        Parameters
        ----------
        n: int
            The number of most relevant pairwise marginals that will be returned.
            Defaults to -1. In this case, all pairwise marginals are returned.

        Returns:
        -------
        list:
            The n most important pairwise marginals.
        """
        self.tot_imp_dict = OrderedDict()
        pairwise_marginals = []

        pairs = list(it.combinations(self.hp_names, 2))
        for combi in pairs:
            pairwise_marginal_performance = self.get_importances(list(combi), depth=2)
            tot_imp = pairwise_marginal_performance[combi][0]
            pairwise_marginals.append((tot_imp, combi[0], combi[1]))

        pairwise_marginal_performance = sorted(pairwise_marginals, reverse=True)

        if n == -1:
            n = len(pairwise_marginal_performance)

        for marginal, p1, p2 in pairwise_marginal_performance[:n]:
            self.tot_imp_dict[(p1, p2)] = marginal

        return self.tot_imp_dict
