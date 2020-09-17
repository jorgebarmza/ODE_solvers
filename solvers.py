import numpy as np
from tabulate import tabulate
import math


def euler(y_prime, x_init, y_init, dx, x_final):
    """Euler's method

    Args:
      y_prime: dy/dx
      x_init: initial value of x
      y_init: initial value of y
      dx: change in x for each itearation.
      x_final: last x value to consider.

    Returns:
      list of (x,y) pairs representing the function y(x).
    """

    timesteps = math.ceil((x_final - x_init) / dx) + 1

    x = np.linspace(x_init, x_final, timesteps)

    y = np.zeros(timesteps, dtype=np.float32)
    y[0] = y_init

    for i in range(1, timesteps):
        y[i] = y[i - 1] + dx * y_prime(x[i - 1], y[i - 1])

    return list(zip(range(timesteps), x, y))


def heun(y_prime, x_init, y_init, dx, x_final):
    """Heun's method

    Args:
      y_prime: dy/dx
      x_init: initial value of x
      y_init: initial value of y
      dx: change in x for each itearation.
      x_final: last x value to consider.

    Returns:
      list of (x,y) pairs representing the function y(x).
    """

    timesteps = math.ceil((x_final - x_init) / dx) + 1

    x = np.linspace(x_init, x_final, timesteps)

    y = np.zeros(timesteps, dtype=np.float32)
    y[0] = y_init

    for i in range(1, timesteps):
        left_tan = y_prime(x[i - 1], y[i - 1])
        y[i] = y[i - 1] + dx * left_tan
        right_tan = y_prime(x[i], y[i])

        y[i] = y[i - 1] + dx / 2 * (left_tan + right_tan)

    return list(zip(range(timesteps), x, y))


def pretty_print(f):
    print(tabulate(f, headers=["i", "x", "y"], floatfmt=".4f"))


if __name__ == "__main__":

    # Given derivative expression.
    def y_prime(x, y):
        return x * math.sqrt(y)

    y_prime = y_prime
    x_init = 1
    y_init = 4
    dx = 0.1
    x_final = 1.5

    print("Euler")
    f1 = euler(y_prime, x_init, y_init, dx, x_final)
    pretty_print(f1)
    print()

    print("Heun")
    f2 = heun(y_prime, x_init, y_init, dx, x_final)
    pretty_print(f2)
    print()
