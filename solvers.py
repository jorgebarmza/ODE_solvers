import numpy as np
from tabulate import tabulate
import math

def euler(y_prime, x_init, y_init, dt, x_final):
    """Euler method
    
    Args:
      y_prime: dy/dx
      x_init: initial value of x
      y_init: initial value of y
      dx: change in x for each itearation.
      x_final: last x value to consider.
    """

    timesteps =  math.ceil((x_final - x_init) / dt) + 1

    x = np.linspace(x_init, x_final, timesteps)

    y = np.zeros(timesteps, dtype=np.float32)
    y[0] = y_init

    for i in range(1, timesteps):
        y[i] = y[i-1] + dt * y_prime(x[i-1], y[i-1])

    return list(zip(range(timesteps), x, y))

def pretty_print(f):
    print(tabulate(f, headers=['i', 'x', 'y'], floatfmt=".4f"))

if __name__ == '__main__':

  # Given derivative expression.
  def y_prime(x, y):
      return 2*x - 3*y + 1

  f = euler(y_prime=y_prime, x_init=1, y_init=4, dt=0.05, x_final=1.2)

  pretty_print(f)