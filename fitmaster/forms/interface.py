from abc import ABC, abstractmethod
import numpy as np

import numpy.typing as npt


class FunctionalFormStrategy(ABC):
    """
    Abstract base class for functional form strategies.

    This class defines the interface for functional form strategies used.
    Subclasses must implement the `func` and `initial_guess` methods.
    """

    @abstractmethod
    def func(
        self, x: npt.NDArray[np.floating | np.integer], *params
    ) -> npt.NDArray[np.floating | np.integer]:
        """
        Calculate the functional form of the model.

        Parameters:
            x (numpy.ndarray): The input data.
            params (tuple): The parameters of the model.

        Returns:
            numpy.ndarray: The calculated functional form of the model.
        """

        pass

    @abstractmethod
    def initial_guess(
        self,
        x: npt.NDArray[np.floating | np.integer],
        y: npt.NDArray[np.floating | np.integer],
    ) -> list[np.floating | np.integer]:
        """
        Return the initial guess for the model parameters.

        Parameters:
            x (numpy.ndarray): The input data.
            y (numpy.ndarray): The target data.

        Returns:
            list[np.floating | np.integer]: The initial guess for the model parameters.
        """

        pass
