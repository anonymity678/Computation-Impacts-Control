import numpy as np


def phi(x):
    return np.kron(x, x)


def phi_sin(x):
    return np.array([x[0] ** 2, x[0] * x[1], x[1] ** 2])


def phi_12(x):
    return np.array(
        [x[0] ** 2, x[0] * x[1], x[1] ** 2, x[0] ** 3, x[0] ** 2 * x[1], x[1] ** 2 * x[0], x[1] ** 3, x[0] ** 4,
         x[0] ** 3 * x[1], x[0] ** 2 * x[1] ** 2, x[1] ** 3 * x[0], x[1] ** 4])


def phi_sinquart(x):
    return np.array(
        [x[0] ** 2, x[0] * x[1], x[1] ** 2, x[0] ** 4, x[0] ** 3 * x[1], x[1] ** 2 * x[0] ** 2, x[0] * x[1] ** 3,
         x[1] ** 4])
