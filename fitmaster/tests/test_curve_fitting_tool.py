import numpy as np
from fitmaster.core.curve_fitting_tool import CurveFittingTool, CurveFittingVisualizer
from fitmaster.forms.concrete import LinearForm


def test_CurveFittingTool():
    tool = CurveFittingTool()

    # Generate some example data
    x = np.linspace(0, 10, 100)
    y = 3 * x + 2 + np.random.normal(0, 1, len(x))

    # Fit and evaluate a linear function
    results = tool.fit_and_evaluate(x, y, "linaer", LinearForm())

    # Check the results
    assert "form" in results
    assert "params" in results
    assert "y_pred" in results

    # Search for the best fit among a list of functional forms and evaluate the fit
    results = tool.search_and_evaluate(x, y, ["linear", "quadratic"])

    # Check the results
    assert isinstance(results, list)
    for result in results:
        assert "form" in result
        assert "params" in result
        assert "y_pred" in result


def test_CurveFittingVisualizer():
    visualizer = CurveFittingVisualizer()
    tool = CurveFittingTool()

    # Generate some example data
    x = np.linspace(0, 10, 100)
    y = 3 * x + 2 + np.random.normal(0, 1, len(x))

    # Search for the best fit among a list of functional forms and evaluate the fit
    results = tool.search_and_evaluate(x, y, ["linear", "quadratic"])

    # Plot the fits
    fig, ax = visualizer.plot_fits(x, y, results, save_fig=False)

    # Check the results
    assert fig is not None
    assert ax is not None

    # Plot the residuals
    figs_axes = visualizer.plot_residuals(x, y, results, save_fig=False)

    # Check the results
    assert isinstance(figs_axes, list)
    for fig, ax in figs_axes:
        assert fig is not None
        assert ax is not None
