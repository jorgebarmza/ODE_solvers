"""Defines functions to approximate ODE solutions."""
import numpy as np
from tabulate import tabulate
import math
import time


def preprocess(x_init, y_init, h, x_final):
    timesteps = math.ceil((x_final - x_init) / h) + 1

    x = np.linspace(x_init, x_final, timesteps)

    y = np.zeros(timesteps, dtype=np.float128)
    y[0] = y_init

    return timesteps, x, y


def euler(f, x_init, y_init, h, x_final):
    """Euler's method

    Args:
      f: dy/dx.
      x_init: initial value of x.
      y_init: initial value of y.
      h: change in x for each itearation.
      x_final: last x value to consider.

    Returns:
      list of (x,y) pairs representing the function y(x).
    """

    timesteps, x, y = preprocess(x_init, y_init, h, x_final)

    for i in range(1, timesteps):
        y[i] = y[i - 1] + h * f(x[i - 1], y[i - 1])

    return list(zip(range(timesteps), x, y))


def heun(f, x_init, y_init, h, x_final):
    """Heun's method

    Args:
      f: dy/dx.
      x_init: initial value of x.
      y_init: initial value of y.
      h: change in x for each itearation.
      x_final: last x value to consider.

    Returns:
      list of (x,y) pairs representing the function y(x).
    """

    timesteps, x, y = preprocess(x_init, y_init, h, x_final)

    for i in range(1, timesteps):
        left_tan = f(x[i - 1], y[i - 1])
        y[i] = y[i - 1] + h * left_tan
        right_tan = f(x[i], y[i])

        y[i] = y[i - 1] + h / 2 * (left_tan + right_tan)

    return list(zip(range(timesteps), x, y))


def rk4(f, x_init, y_init, h, x_final):
    """Standard Runge-Kunta's method

    Args:
      f: dy/dx = f(x,y).
      x_init: initial value of x.
      y_init: initial value of y.
      h: change in x for each itearation.
      x_final: last x value to consider.

    Returns:
      list of (x,y) pairs representing the function y(x).
    """

    timesteps, x, y = preprocess(x_init, y_init, h, x_final)

    for i in range(1, timesteps):
        k1 = f(x[i - 1], y[i - 1])
        k2 = f(x[i - 1] + h / 2, y[i - 1] + h * k1 / 2)
        k3 = f(x[i - 1] + h / 2, y[i - 1] + h * k2 / 2)
        k4 = f(x[i - 1] + h, y[i - 1] + h * k3)

        y[i] = y[i - 1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return list(zip(range(timesteps), x, y))


def solution_to_str(solution, decimals):
    """Convers ODE solution to string."""
    return tabulate(solution,
                    headers=["i", "x", "y"],
                    floatfmt=".{}f".format(decimals))


def write(filename, solution, decimals):
    """Writes ODE solution to a file."""
    fi = open(filename, 'w')
    fi.write(solution_to_str(solution, decimals))
    fi.close()


def approximate(filename, decimals, target, method, n_init, f, x_init, y_init,
                x_final):
    """Approximates an ODE solution and writes it to a file.
    
    The number of subintervals, n, is duplicated until the desired number of
    decimals precision is obtained.
    """
    n = n_init
    while True:
        # Get approximation
        h = (x_final - x_init) / n
        solution = method(f, x_init, y_init, h, x_final)
        approx = solution[len(solution) - 1][2]
        if str(round(approx, decimals)) == str(round(target, decimals)):
            break
        n *= 2
    write(filename, solution, decimals)