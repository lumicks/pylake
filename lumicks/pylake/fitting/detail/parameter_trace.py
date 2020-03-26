from .utilities import parse_transformation
import numpy as np
import scipy.optimize as optim


def parameter_trace(model, parameters, inverted_parameter, independent, dependent, **kwargs):
    """Invert a model with respect to one parameter. This function fits a unique parameter for every data point in
    this data-set while keeping all other parameters fixed. This can be used to for example invert the model with
    respect to the contour length or some other parameter.

    Parameters
    ----------
    model : Model
        Fitting model.
    parameters : Parameters
        Model parameters.
    inverted_parameter : str
        Parameter to invert.
    independent : array_like
        vector of values for the independent variable
    dependent: array_like
        vector of values for the dependent variable
    **kwargs:
        forwarded to scipy.optimize.least_squares

    Examples
    --------
    ::
        # Define the model to be fitted
        model = pylake.inverted_odijk("model") + pylake.offset("f", "offset")

        # Fit the overall model first
        data_handle = model.load_data(distance, force)
        current_fit = pylake.FitObject(model)
        current_fit.fit()

        # Calculate a per data point contour length
        lcs = parameter_trace(model, current_fit.parameters[data_handle], "model_Lc", distance, force)
    """
    parameter_names = model.parameter_names
    assert inverted_parameter in parameters, f"Inverted parameter not in model parameter vector {parameters}."
    for key in parameter_names:
        assert key in parameters, f"Missing parameter {key} in supplied parameter vector."

    # Grab reference parameter vector and index for the parameter list
    parameter_vector = [parameters[key].value for key in parameter_names]
    lb = parameters[inverted_parameter].lb
    ub = parameters[inverted_parameter].ub
    inverted_parameter_index = parameter_names.index(inverted_parameter)

    def fit_single_point(x, y):
        def residual(inverted_parameter_value):
            parameter_vector[inverted_parameter_index] = inverted_parameter_value
            return y - model._raw_call(np.array([x]), parameter_vector)

        def jacobian(inverted_parameter_value):
            parameter_vector[inverted_parameter_index] = inverted_parameter_value
            return -model.jacobian(np.array([x]), parameter_vector)[inverted_parameter_index, :]

        jac = jacobian if model.has_jacobian else "2-point"
        result = optim.least_squares(residual, parameter_vector[inverted_parameter_index], jac=jac,
                                     bounds=(lb, ub), method='trf', **kwargs)

        return result.x[0]

    return np.array([fit_single_point(x, y) for (x, y) in zip(independent, dependent)])
