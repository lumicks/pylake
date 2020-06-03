import warnings
import numpy as np
import scipy.optimize as optim


def numerical_diff(fn, x, dx=1e-6):
    return (fn(x + dx) - fn(x - dx)) / (2.0 * dx)


def numerical_jacobian(fn, parameter_vector, dx=1e-6):
    finite_difference_jacobian = np.zeros((len(parameter_vector), len(fn(parameter_vector))))
    for i in np.arange(len(parameter_vector)):
        params = np.copy(parameter_vector)
        params[i] = params[i] + dx
        up = fn(params)
        params[i] = params[i] - 2.0 * dx
        down = fn(params)
        finite_difference_jacobian[i, :] = (up - down) / (2.0*dx)

    return finite_difference_jacobian


def inversion_functions(model_function, f_min, f_max, derivative_function, tol):
    """This function generates two functions which allow for inverting models via optimization. These functions are used
    by functions which require inversion of a model's dependent and independent variable"""

    def fit_single(single_distance, initial_guess):
        """Invert a single independent / dependent data point"""
        jac = derivative_function if derivative_function else "2-point"
        single_estimate = optim.least_squares(lambda f: model_function(f) - single_distance, initial_guess, jac=jac,
                                              bounds=(f_min, f_max), method='trf', ftol=tol, xtol=tol, gtol=tol)

        return single_estimate.x[0]

    def manual_inversion(distances, initial_guess):
        """Invert the dependent and independent variable for a list"""
        return np.array([fit_single(distance, initial_guess) for distance in distances])

    return manual_inversion, fit_single


def invert_function(d, initial, f_min, f_max, model_function, derivative_function=None, tol=1e-8):
    """This function inverts a function using a least squares optimizer. For models where this is required, this is the
    most time consuming step.

    Parameters
    ----------
    d : array_like
        old independent parameter
    initial : float
        initial guess for the optimization procedure
    f_min : float
        minimum bound for inverted parameter
    f_max : float
        maximum bound for inverted parameter
    model_function : callable
        non-inverted model function
    derivative_function : callable
        model derivative with respect to the independent variable (returns an element per data point)
    tol : float
        optimization tolerances
    """
    manual_inversion, _ = inversion_functions(model_function, f_min, f_max, derivative_function, tol)

    return manual_inversion(d, initial)


def invert_jacobian(d, inverted_model_function, jacobian_function, derivative_function):
    """This function computes the jacobian of the model when the model has been inverted with respect to the independent
    variable.

    The Jacobian of the function with one variable inverted is related to the original Jacobian. This transformation
    can be derived from the implicit function:

    G = F - f_inverse(f(F, p), p) = 0

    Differentiation w.r.t. p_i yields:

    0 = δG / δp_i + (δG / δf_inverse(f(F, p), p)) * (δf_inverse(f(F, p), p) / δp_i)

    0 = (δG / δf_inverse(f(F, p), p)) * (δf_inverse(f(F, p), p) / δp_i)

    0 = - δf_inverse(f(F, p), p) / δp_i
      = (δf_inverse(f(F, p), p) / δf(F, p)) (δf(F, p) / δp_i) + δf_inverse(f(F, p), p) / δp_i
      = (δf_inverse(dist, p) / δdist) (δf(F, p) / δp_i) + δf_inverse(dist, p) / δp_i

    Hence:
        δf_inverse(dist, p) / δp_i = - (δf_inverse(dist, p) / δdist) (δf(F, p) / δp_i)

    Since (see invert_derivative)
        δf_inverse(dist, p) / δdist = ( δf(F, p) / δF )^-1

    We obtain:
        δf_inverse(dist, p) / δp_i = - (δf(F, p) / δp_i) ( δf(F, p) / δF )^-1

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


def invert_derivative(d, inverted_model_function, derivative_function):
    """
    Calculates the derivative of the inverted function.

    F = f_inverse(f(F), p)

    Derive both sides w.r.t. F gives:

    1 = δf_inverse(f(F), p) / δF

    or:

    1 = ( δf_inverse(f(F), p) / δf(F) ) ( δf(F) / δF )

    or:

    δf_inverse(d, p) / δd = ( δf(F) / δF )^-1

    Parameters
    ----------
    d : values for the old independent variable
    inverted_model_function : callable
        inverted model function (model with the dependent and independent variable exchanged)
    derivative_function : callable
        derivative of the non-inverted model w.r.t. the independent variable
    """
    return 1.0 / derivative_function(inverted_model_function(d))
