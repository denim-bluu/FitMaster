from typing import Any
import numpy as np
from .interface import ModelSelectionCriterionStrategy

import numpy.typing as npt


class AICCriterion(ModelSelectionCriterionStrategy):
    def evaluate(
        self,
        y: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        num_params: int,
    ) -> np.floating[Any]:
        """Evaluate the Akaike Information Criterion (AIC) value.

        Args:
            y: The actual values.
            y_pred: The predicted values.
            num_params: The number of parameters.

        Returns:
            The AIC value.
        """
        resid = y - y_pred
        sse = np.sum(resid**2)
        return 2 * num_params + len(y) * np.log(sse / len(y))


class BICCriterion(ModelSelectionCriterionStrategy):
    def evaluate(
        self,
        y: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        num_params: int,
    ) -> np.floating[Any]:
        """Evaluate the Bayesian Information Criterion (BIC) value.

        Args:
            y: The actual values.
            y_pred: The predicted values.
            num_params: The number of parameters.

        Returns:
            The BIC value.
        """
        resid = y - y_pred
        sse = np.sum(resid**2)
        return num_params * np.log(len(y)) + len(y) * np.log(sse / len(y))


class RSquaredCriterion(ModelSelectionCriterionStrategy):
    def evaluate(
        self,
        y: npt.NDArray[np.floating],
        y_pred: npt.NDArray[np.floating],
        num_params: int,
    ) -> np.floating[Any]:
        """Evaluate the R-squared value.

        Args:
            y: The actual values.
            y_pred: The predicted values.
            num_params: The number of parameters (not used).

        Returns:
            The R-squared value.
        """
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        return 1 - (ss_res / ss_total)
