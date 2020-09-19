import numpy as np
from tabulate import tabulate
import math
import time


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


def f_to_str(f, decimals):
    return tabulate(f,
                    headers=["i", "x", "y"],
                    floatfmt=".{}f".format(decimals))


def write(filename, f, decimals):
    fi = open(filename, 'w')
    fi.write(f_to_str(f, decimals))
    fi.close()


def approximate(filename, decimals, target, method, n_init, y_prime, x_init,
                y_init, x_final):
    n = n_init
    while True:
        # Get approximation
        dx = (x_final - x_init) / n
        f = method(y_prime, x_init, y_init, dx, x_final)
        approx = f[len(f) - 1][2]
        if str(round(approx, decimals)) == str(round(target, decimals)):
            break
        n *= 2
    write(filename, f, decimals)


if __name__ == "__main__":

    # Given derivative expression.
    def y_prime(x, y):
        return y

    p1_euler_dict = dict(
        filename='psd1_euler.txt',
        decimals=3,
        target=2.71828182846,
        method=euler,
        n_init=50,
        y_prime=y_prime,
        x_init=0,
        y_init=1,
        x_final=1,
    )

    # p1_heun_dict = dict(
    #     filename='p1_euler.txt',
    #     decimals=3,
    #     target=2.71828182846,
    #     method=euler,
    #     n_init=50,
    #     y_prime=y_prime,
    #     x_init=0,
    #     y_init=1,
    #     x_final=1,
    # )

    approximate(**p1_euler_dict)
