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
MAX_STEPS_PER_LOOP: int = 2500  # A safeguard to slow down the program if it's running too slow
simulation_time_s: float = 0  # The time, in seconds, since the simulation began
start_time_s: float = time.perf_counter()
SIMULATION_SPEED: float = 0.1

# Vectors
VECTOR_SPACING: float = 0.5  # Less spacing means the vectors will be closer packed together (more detail, hard to see)
VERTICAL_VECTOR_MIN: float = -10
VERTICAL_VECTOR_MAX: float = 10

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

    vector_steps: int = int(x_range/VECTOR_SPACING)
    x_vector: npt.NDArray[np.float64] = np.linspace(HORIZONTAL_GRAPH_MIN, HORIZONTAL_GRAPH_MAX, vector_steps)
    y_vector: npt.NDArray[np.float64] = np.linspace(VERTICAL_VECTOR_MIN, VERTICAL_VECTOR_MAX, vector_steps)
    x_vector, y_vector = np.meshgrid(x_vector, y_vector)
    dx_vector: npt.NDArray[np.float64] = y_vector
    dy_vector: npt.NDArray[np.float64] = differential_eq(x_vector, y_vector)
    m_vector: npt.NDArray[np.float64] = np.sqrt(dx_vector**2 + dy_vector**2)
    d_vector: npt.NDArray[np.float64] = np.arctan2(dy_vector, dx_vector)
    dx_vector = np.cos(d_vector)*VECTOR_SPACING
    dy_vector = np.sin(d_vector)*VECTOR_SPACING

    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title("ZanderPeterson/ModelingPointsOnGraph-ODE")
    ax.plot(x_points, y_points, color="white")
    ax.set_xlabel("X Values")
    ax.set_ylabel("Y Values")
    ax.set_title("Main Function")
    ax.grid(True, alpha=0.4)

    # Main loop
    a_points, = plt.plot(0, 0, 'r')
    v_points, = plt.plot(0, 0, 'r')
    vector_plot = plt.quiver(x_vector, y_vector, dx_vector, dy_vector, m_vector,
                             cmap='viridis', angles='xy', scale_units='xy', scale=1)
    colorbar = fig.colorbar(vector_plot)
    time.sleep(1)  # Improves timing by giving the program a second to finish loading
    start_time_s: float = time.perf_counter()
    while plt.fignum_exists(fig.number):
        current_time_s: float = time.perf_counter()
        required_steps: int = int((current_time_s - simulation_time_s*SIMULATION_SPEED - start_time_s)
                                  / SIMULATION_TIME_STEP)
        if required_steps >= MAX_STEPS_PER_LOOP:
            print(f"Warning, hit step maximum ({required_steps} required vs max of {MAX_STEPS_PER_LOOP})")
            required_steps = 1000
        simulation_time_s += required_steps*SIMULATION_TIME_STEP/SIMULATION_SPEED
        for i in range(required_steps):
            simulated_points[:, 2] = differential_eq(simulated_points[:, 0], simulated_points[:, 1])
            simulated_points[:, 0] += simulated_points[:, 1]*SIMULATION_TIME_STEP*SIMULATION_SPEED
            simulated_points[:, 1] += simulated_points[:, 2]*SIMULATION_TIME_STEP*SIMULATION_SPEED

        a_points.remove()
        v_points.remove()
        a_points, = plt.plot(simulated_points[:, 0], main_function(simulated_points[:, 0]), 'ro')
        v_points, = plt.plot(simulated_points[:, 0], simulated_points[:, 1], 'go')
        plt.pause(0.05)
    print("Figure closed")
