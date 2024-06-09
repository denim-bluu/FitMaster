import numpy as np
from .interface import FunctionalFormStrategy

import numpy.typing as npt


class LinearForm(FunctionalFormStrategy):
    """
    Represents a linear functional form.

    Args:
        x (npt.NDArray[np.floating | np.integer]): The input array.
        a (float): The coefficient of the linear term.
        b (float): The coefficient of the constant term.

    Returns:
        npt.NDArray[np.floating | np.integer]: The output array.

    """

    def func(
        self,
        x: npt.NDArray[np.floating | np.integer],
        a: float,
        b: float,
    ) -> npt.NDArray[np.floating | np.integer]:
        return a + b * x

    def initial_guess(
        self,
        x: npt.NDArray[np.floating | np.integer],
        y: npt.NDArray[np.floating | np.integer],
    ):
        """
        Provides an initial guess for the linear form parameters.

        Args:
            x (npt.NDArray[np.floating | np.integer]): The input array.
            y (npt.NDArray[np.floating | np.integer]): The output array.

        Returns:
            List[Any]: The initial guess for the parameters.

        """
        return [1, 1]


class ExponentialForm(FunctionalFormStrategy):
    """
    Represents an exponential functional form.

    Args:
        x (npt.NDArray[np.floating | np.integer]): The input array.
        a (float): The coefficient of the exponential term.
        b (float): The exponent of the exponential term.
        c (float): The constant term.

    Returns:
        npt.NDArray[np.floating | np.integer]: The output array.

    """

    def func(
        self,
        x: npt.NDArray[np.floating | np.integer],
        a: float,
        b: float,
        c: float,
    ) -> npt.NDArray[np.floating | np.integer]:
        return a * np.exp(b * x) + c

    def initial_guess(
        self,
        x: npt.NDArray[np.floating | np.integer],
        y: npt.NDArray[np.floating | np.integer],
    ):
        """
        Provides an initial guess for the exponential form parameters.

        Args:
            x (npt.NDArray[np.floating | np.integer]): The input array.
            y (npt.NDArray[np.floating | np.integer]): The output array.

        Returns:
            List[Any]: The initial guess for the parameters.

        """
        return [1, 1, 1]


class LogarithmicForm(FunctionalFormStrategy):
    """
    Represents a logarithmic functional form.

    Args:
        x (npt.NDArray[np.floating | np.integer]): The input array.
        a (float): The coefficient of the logarithmic term.
        b (float): The constant term.

    Returns:
        npt.NDArray[np.floating | np.integer]: The output array.

    """

    def func(
        self,
        x: npt.NDArray[np.floating | np.integer],
        a: float,
        b: float,
    ) -> npt.NDArray[np.floating | np.integer]:
        return a + b * np.log(x)

    def initial_guess(
        self,
        x: npt.NDArray[np.floating | np.integer],
        y: npt.NDArray[np.floating | np.integer],
    ):
        """
        Provides an initial guess for the logarithmic form parameters.

        Args:
            x (npt.NDArray[np.floating | np.integer]): The input array.
            y (npt.NDArray[np.floating | np.integer]): The output array.

        Returns:
            List[Any]: The initial guess for the parameters.

        """
        return [1, 1]
