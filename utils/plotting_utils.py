import numpy as np
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from typing import Tuple, Union, List


def plot_var(
      variable: np.ndarray,
      easting: np.ndarray,
      northing: np.ndarray, 
      var_name: str, 
      clims: Union[
         Tuple[Union[int, float], ...], 
         List[Union[int, float]]
      ],
      pi_cbar: bool =False,
      cmap: str = 'viridis'
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
        northing.max(),northing.min()
    ]
    plt.imshow(variable, extent=extent, aspect='auto', cmap=cmap)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.title(var_name, fontsize=10)
    plt.xlabel('UTM Zone 13 Easting [m]', fontsize=10)
    plt.ylabel('UTM Zone 13 Northing [m]', fontsize=10)
    plt.clim(clims[0], clims[1])

    if pi_cbar:
      cbar=plt.colorbar()
      cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
      cbar.set_ticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
    else:
      plt.colorbar()

    current_figure = plt.gcf()
    return current_figure
