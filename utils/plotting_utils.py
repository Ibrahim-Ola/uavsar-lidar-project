import numpy as np
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from typing import Tuple, Union, List


def plot_var(
      variable: np.ndarray,
      easting: np.ndarray,
      northing: np.ndarray, 
      clims: Union[
         Tuple[Union[int, float], ...], 
         List[Union[int, float]]
      ],
      pi_cbar: bool =False,
      cmap: str = 'viridis',
      var_name: str = 'Variable'
) -> Figure:
    """
    A function that Plots a variable/remote sensing parameters with a colorbar and
    axis labels on a 2D grid. The colorbar limits are set by the user. The x and y axis
    are set to the easting and northing of the raster (UAVSAR) data.

    Parameters
    ----------
    variable : 2D numpy array
        A two-dimensional array representing the variable to be plotted.
    easting : 2D numpy array
        A two-dimensional array representing the easting of the raster data.
    northing : 2D numpy array
        A two-dimensional array representing the northing of the raster data.
    var_name : str
        A string representing the title of the plot, typically the name of the variable.
    clims : tuple or list of two elements
        A tuple or list containing two elements representing the minimum and maximum values of the color scale.

    Returns
    -------
    current_figure : matplotlib Figure object

    The function generates a plot using matplotlib, with UTM Zone 13 Easting and Northing as axes.
    The input variable is represented as color intensity on the plot. The color scale is controlled
    by the 'clims' parameter

    Example
    -------
    >>> plot_var(np.array([[1,2], [3,4]]), 'Example Variable', (1, 4))
    """


    extent=[
        easting.min(), easting.max(),
        northing.max(), northing.min()
    ]
    plt.imshow(variable, extent=extent, aspect='auto', cmap=cmap)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.xlabel('Easting [m]', fontsize=18)
    plt.xticks(fontsize=14)
    plt.ylabel('Northing [m]', fontsize=18)
    plt.yticks(fontsize=14)
    plt.clim(clims[0], clims[1])

    if pi_cbar:
        cbar = plt.colorbar()
        cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
        cbar.ax.tick_params(labelsize=18)  # Increase tick size for colorbar
        cbar.set_label(label=var_name, fontsize=18)
    else:
        cbar=plt.colorbar()
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(label=var_name, fontsize=18)

    current_figure = plt.gcf()

    return current_figure


def plot_results(ax, X, y, xlabel, ylabel, r2_value, rmse_value, pos1=0.8, pos2=1.6):
    """
    Plot the results of a regression model.

    Parameters:
    - ax: Matplotlib axis object. The axis on which to plot the results.
    - X: array-like, shape (n_samples,). Predictive values used for the x-axis in the plot.
    - y: array-like, shape (n_samples,). Observed values used for the y-axis in the plot.
    - predictions: array-like, shape (n_samples,). Predicted values from the model.
    - title: str. Title for the plot.
    - xlabel: str. Label for the x-axis.
    - ylabel: str. Label for the y-axis.
    - r2_value: float. The R-squared value for the model's predictions.
    - pos1: float, optional. Horizontal position for the R-squared annotation (default is 0.8).

    Returns:
    None
    """
    ax.scatter(X, y, edgecolor='black', s=100)
    ax.plot([0,1], [0,1], transform=ax.transAxes, color='black', lw=2)
    ax.text(pos1, pos2, f'$R^2 = {r2_value:.3f}$\nRMSE = {rmse_value*100:.3f} [cm]', fontsize=20)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.tick_params(axis='x', labelsize=18)  
    ax.tick_params(axis='y', labelsize=18)  
    ax.set_xlabel(xlabel, fontsize=24)
