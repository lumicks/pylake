import numpy as np
import scipy as sp
import scipy.optimize as optim


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


def invert_function(d, initial, f_min, f_max, model_function, derivative_function=None):
    """This function inverts a function using a least squares optimizer. For models where this is required, this is the
    most time consuming step.

    Parameters
    ----------
    d : array_like
        old independent parameter
    initial : array_like
        initial guess for the optimization procedure
    f_min : float
        minimum bound for inverted parameter
    f_max : float
        maximum bound for inverted parameter
    model_function : callable
        non-inverted model function
    derivative_function : callable
        model derivative with respect to the independent variable (returns an element per data point)
    """
    def jacobian(f_trial):
        return sp.sparse.diags(derivative_function(f_trial), offsets=0)

    jac = jacobian if derivative_function else "2-point"

    result = optim.least_squares(lambda f_trial: model_function(f_trial) - d, initial, jac=jac,
                                 jac_sparsity=sp.sparse.identity(len(d)),
                                 bounds=(f_min, f_max), method='trf', ftol=1e-06, xtol=1e-08, gtol=1e-8)

    return result.x


def invert_jacobian(d, inverted_model_function, jacobian_function, derivative_function):
    """This function computes the jacobian of the model when the model has been inverted with respect to the independent
    variable.

    The Jacobian of the function with one variable inverted is related to the original Jacobian
    The transformation Jacobian is structured as follows:

    [  dy/dF   dy/db   dy/dc  ]
    [   0        1       0    ]
    [   0        0       1    ]

    The inverse of this Jacobian provides us with the actual parameters that we are interested in. It is given by:
    [ (dy/da)^-1  -(dy/db)(dy/dF)^-1    -(dy/dc)(dy/dF)^-1 ]
    [    0                1                     0          ]
    [    0                0                     1          ]

    Parameters
    ----------
    d : values for the old independent variable
    inverted_model_function : callable
        inverted model function (model with the dependent and independent variable exchanged)
    jacobian_function : callable
        derivatives of the non-inverted model
    derivative_function : callable
        derivative of the non-inverted model w.r.t. the independent variable
    """
    F = inverted_model_function(d)
    jacobian = jacobian_function(F)
    derivative = derivative_function(F)
    inverse = 1.0/derivative
    inverted_dyda = np.tile(inverse, (jacobian.shape[0], 1))
    jacobian = -jacobian * inverted_dyda

    return jacobian
