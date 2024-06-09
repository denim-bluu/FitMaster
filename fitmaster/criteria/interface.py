from abc import ABC, abstractmethod
import numpy.typing as npt
import numpy as np
from typing import Any


class ModelSelectionCriterionStrategy(ABC):
    """
    Interface for model selection criterion strategies.

    This interface defines the methods that should be implemented by any model selection criterion strategy.
    Subclasses should implement the `evaluate` method to evaluate the model selection criterion.

    Attributes:
        None

    Methods:
        evaluate: Evaluates the model selection criterion for a given set of predictions.

    """

    @abstractmethod
    def evaluate(
        self,
        y: npt.NDArray[np.floating | np.integer],
        y_pred: npt.NDArray[np.floating | np.integer],
        num_params: int,
    ) -> np.floating[Any]:
        """
        Evaluates the model selection criterion for a given set of predictions.

        Args:
            y (npt.NDArray[np.floating | np.integer]): The true values of the target variable.
            y_pred (npt.NDArray[np.floating | np.integer]): The predicted values of the target variable.
            num_params (int): The number of parameters in the model.

        Returns:
            np.floating[Any]: The value of the model selection criterion.

        """
        pass
