# Standard Libraries
from collections.abc import Callable
import math
import time

# Third-Party Libraries
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

# Functions (Can be changed, but must be ufunc (numpy arrays can be passed through it))
main_function: np.ufunc = lambda x: (np.sin(x))
main_derivative: np.ufunc = lambda x: (np.cos(x))

FRICTION: float = 0.8
GRAVITY: float = 9.81
differential_eq: np.ufunc = lambda x, v: (-v*FRICTION - GRAVITY*np.sin(np.atan(main_derivative(x))))

# Points (Written as an array of [x, v] pairs, each being a point [[x, v, a], ..., [x, v, a]])
# Note: a will be overwritten and thus will not change anything.
simulated_points: npt.NDArray[np.float64] = np.array([[0.0, 0.0, 0.0],
                                                      [-1.0, 10.0, 0.0]])
SIMULATION_TIME_STEP: float = 0.0001  # Smaller steps in time allows for more accuracy, at the cost of processing power
simulation_time_s: float = 0  # The time, in seconds, since the simulation began
start_time_s: float = time.perf_counter()

# Constants pertaining to drawing the continuous function
FUNCTION_DETAIL: float = 0.001
HORIZONTAL_GRAPH_MIN: float = -10
HORIZONTAL_GRAPH_MAX: float = 10

if __name__ == "__main__":
    print("Running main2d.py")

    plt.style.use('dark_background')  # Sets the theme
    plt.ion()  # Allows for interactions w/loops

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

    # Main loop
    points, = plt.plot(0, 0, 'r')
    while plt.fignum_exists(fig.number):
        current_time_s: float = time.perf_counter()
        while (current_time_s - simulation_time_s) > start_time_s:
            simulation_time_s += SIMULATION_TIME_STEP
            simulated_points[:, 2] = differential_eq(simulated_points[:, 0], simulated_points[:, 1])
            simulated_points[:, 0] += simulated_points[:, 1]*SIMULATION_TIME_STEP
            simulated_points[:, 1] += simulated_points[:, 2]*SIMULATION_TIME_STEP

        points.remove()
        points, = plt.plot(simulated_points[:, 0], main_function(simulated_points[:, 0]), 'ro')
        plt.pause(0.05)
    print("Figure closed")
