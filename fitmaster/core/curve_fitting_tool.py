import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import numpy.typing as npt
import scipy.stats as stats
from scipy.optimize import curve_fit

from fitmaster.criteria.factory import ModelSelectionCriterionFactory
from fitmaster.forms.factory import FunctionalFormFactory
from fitmaster.forms.interface import FunctionalFormStrategy


class CurveFittingTool:
    def __init__(self):
        """
        Initialize the CurveFittingTool with factories for functional forms and model selection criteria.
        """
        self.form_factory = FunctionalFormFactory()
        self.criterion_factory = ModelSelectionCriterionFactory()

    def fit_and_evaluate(
        self,
        x: npt.NDArray[np.floating | np.integer],
        y: npt.NDArray[np.floating | np.integer],
        form: str,
        funtional_form: FunctionalFormStrategy,
        criterions: list[str] | None = None,
        **kwargs,
    ):
        """
        Fit a curve to the data and evaluate the fit.

        Parameters:
        x (npt.NDArray): The x data.
        y (npt.NDArray): The y data.
        form (str): The form of the function to fit.
        f (function): The function to fit.
        criterions (list[str], optional): The criteria to use for evaluating the fit.

        Returns:
        dict: A dictionary with the results of the fit and evaluation.
        """
        params, _ = curve_fit(
            f=funtional_form.func,
            xdata=x,
            ydata=y,
            p0=funtional_form.initial_guess(x, y),
            **kwargs,
        )
        y_pred = funtional_form.func(x, *params)

        criteria_results = {
            name: criterion.evaluate(y, y_pred, len(params))
            for name, criterion in self.criterion_factory.criterions.items()
            if criterions is None or name in criterions
        }

        return {
            "form": form,
            "params": params,
            "y_pred": y_pred,
            **criteria_results,
        }

    def search_and_evaluate(
        self,
        x: npt.NDArray[np.floating | np.integer],
        y: npt.NDArray[np.floating | np.integer],
        functional_forms: list[str] | None = None,
        criterions: list[str] | None = None,
        **kwargs,
    ):
        """
        Search for the best fit among a list of functional forms and evaluate the fit.

        Parameters:
        x (npt.NDArray): The x data.
        y (npt.NDArray): The y data.
        functional_forms (list[str], optional): The functional forms to consider.
        criterions (list[str], optional): The criteria to use for evaluating the fit.

        Returns:
        list[dict]: A list of dictionaries with the results of the fits and evaluations, sorted by R^2 value.
        """
        results = [
            self.fit_and_evaluate(x, y, form, f, criterions, **kwargs)
            for form, f in self.form_factory.functional_forms.items()
            if functional_forms is None or form in functional_forms
        ]

        results.sort(key=lambda result: result["r_squared"], reverse=True)
        return results


class CurveFittingVisualizer:
    @staticmethod
    def plot_fits(
        x: npt.NDArray[np.floating | np.integer],
        y: npt.NDArray[np.floating | np.integer],
        results: list[dict[str, np.ndarray]],
        save_path: str = "./image/",
        save_fig: bool = True,
    ) -> tuple[Figure, Axes]:
        """
        Plot the fits to the data.

        Parameters:
        x (npt.NDArray): The x data.
        y (npt.NDArray): The y data.
        results (list[dict]): The results of the fits.
        save_path (str): The path to save the figure.
        save_fig (bool): Whether to save the figure.

        Returns:
        tuple[Figure, Axes]: The figure and axes of the plot.
        """
        fig, ax = plt.subplots()
        ax.scatter(x, y, label="Data")
        for result in results:
            ax.plot(
                x,
                result["y_pred"],
                label=f"{result['form']} (R^2={result['r_squared']:.3f})",
            )
        ax.legend()
        if save_fig:
            fig.savefig("{save_path}/fit_results.png")
        return fig, ax

    @staticmethod
    def plot_residuals(
        x: npt.NDArray[np.floating | np.integer],
        y: npt.NDArray[np.floating | np.integer],
        results: list[dict[str, np.ndarray]],
        save_path: str = "./image/",
        save_fig: bool = True,
    ) -> list[tuple[Figure, Axes]]:
        """
        Plot the residuals of the fits.

        Parameters:
        x (npt.NDArray): The x data.
        y (npt.NDArray): The y data.
        results (list[dict]): The results of the fits.
        save_path (str): The path to save the figures.
        save_fig (bool): Whether to save the figure.

        Returns:
        list[tuple[Figure, Axes]]: A list of figures and axes of the plots.
        """
        figs_axes = []
        for result in results:
            residuals = y - result["y_pred"]

            fig, ax = plt.subplots()
            ax.scatter(x, residuals)
            ax.hlines(0, min(x), max(x), colors="r", linestyles="dashed")
            ax.set_title(f'Residuals for {result["form"]}')
            fig.savefig(f"{save_path}/residuals_{result['form']}.png")
            if save_fig:
                figs_axes.append((fig, ax))

            fig_qq, ax_qq = plt.subplots()
            stats.probplot(residuals, dist="norm", plot=ax_qq)
            ax_qq.set_title(f'QQ Plot for {result["form"]}')

            fig_qq.savefig(f"{save_path}/qqplot_{result['form']}.png")
            if save_fig:
                figs_axes.append((fig_qq, ax_qq))

        return figs_axes
