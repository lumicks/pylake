import numpy as np


def numerical_diff(fn, x, dx=1e-6):
    return (fn(x + dx) - fn(x - dx)) / (2.0 * dx)


def numerical_jacobian(fn, parameter_vector, dx=1e-6):
    finite_difference_jacobian = np.zeros((len(parameter_vector), len(fn(parameter_vector))))
    for i in np.arange(len(parameter_vector)):
        parameters = np.copy(parameter_vector)
        parameters[i] = parameters[i] + dx
        up = fn(parameters)
        parameters[i] = parameters[i] - 2.0 * dx
        down = fn(parameters)
        finite_difference_jacobian[i, :] = (up - down) / (2.0*dx)

    return finite_difference_jacobian