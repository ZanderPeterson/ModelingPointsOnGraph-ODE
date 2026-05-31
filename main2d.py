# Standard Libraries
from collections.abc import Callable
import math

# Third-Party Libraries
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

# Functions (Can be changed, but must be ufunc (numpy arrays can be passed through it))
main_function: np.ufunc = lambda x: (np.sin(x))

# Constants pertaining to drawing the continuous function
FUNCTION_DETAIL: float = 0.01
HORIZONTAL_GRAPH_MIN: float = -5
HORIZONTAL_GRAPH_MAX: float = 5

if __name__ == "__main__":
    print("Running main2d.py")

    plt.style.use('dark_background')  # Sets the theme

    x_range: float = HORIZONTAL_GRAPH_MAX - HORIZONTAL_GRAPH_MIN
    x_steps: int = int(x_range/FUNCTION_DETAIL)
    x_points: npt.NDArray[np.float64] = np.linspace(HORIZONTAL_GRAPH_MIN, HORIZONTAL_GRAPH_MAX, x_steps)
    y_points: npt.NDArray[np.float64] = main_function(x_points)

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("ZanderPeterson/ModelingPointsOnGraph-ODE")
    ax.plot(x_points, y_points, color="white")
    ax.set_xlabel("X Values")
    ax.set_ylabel("Y Values")
    ax.set_title("Main Function")
    ax.grid(True, alpha=0.4)
    plt.show()
