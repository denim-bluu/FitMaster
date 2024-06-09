import numpy as np
from fitmaster.core.curve_fitting_tool import CurveFittingTool, CurveFittingVisualizer
import matplotlib.pyplot as plt

# Generate example univariate data
x_data = np.linspace(1, 10, 10)
y_data = 2 * np.exp(1.5 * x_data) + 5 + np.random.normal(0, 5, len(x_data))

# Create the curve fitting tool
tool = CurveFittingTool()
visuals = CurveFittingVisualizer()

# Fit and evaluate models
results = tool.search_and_evaluate(x_data, y_data)

# Print results
for result in results:
    print(result)

# Plot the results
fig, ax = visuals.plot_fits(x_data, y_data, results)
plt.show()

# Plot residuals
figs_axes = visuals.plot_residuals(x_data, y_data, results)
for fig, ax in figs_axes:
    plt.show()
