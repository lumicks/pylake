import numpy as np
import scipy


def parameter_trace(model, params, inverted_parameter, independent, dependent, **kwargs):
    """Fit a model with respect to one parameter for each data point.

    This function fits a unique parameter value for every data point in this data-set while keeping
    all other parameters fixed. This can be used to for example invert the model with respect to
    the contour length or some other parameter.

    Parameters
    ----------
    model : Model
        Fitting model.
    params : Params
        Model parameters.
    inverted_parameter : str
        Parameter to invert.
    independent : array_like
        Vector of values for the independent variable
    dependent : array_like
        Vector of values for the dependent variable
    **kwargs :
        Forwarded to scipy.optimize.least_squares

    Returns
    -------
    parameter_trace : np.ndarray
        Array of fitted parameter values for the parameter being fitted.

    Raises
    ------
    ValueError
        If specifying a parameter to invert over (`inverted_parameter`) that is not part of the
        model.
    ValueError
        If a parameter required for model simulation is missing from the supplied parameters in
        `params`.
    ValueError
        If parameters are provided that do not have a `lower_bound` or `upper_bound` property.

    Examples
    --------
    ::

        # Define the model to be fitted
        model = pylake.ewlc_odijk_force("model") + pylake.force_offset("model")

        # Fit the overall model first
        model.add_data("dataset1", f=force_data, d=distance_data)
        current_fit = pylake.FdFit(model)
        data_handle = current_fit.add_data("my data", force, distance)
        current_fit.fit()

        # Calculate a per data point contour length
        lcs = parameter_trace(model, current_fit[data_handle], "model/Lc", distance, force)

        # Alternatively, if rather than a reference curve to fit, you have model parameters, you
        # can use a Params dictionary obtained from a model directly.
        model2 = lk.ewlc_odijk_distance("m")

        model2["m/Lp"].value = 50
        model2["m/St"].value = 1200
        model2["m/Lc"].value = 5

        lk.parameter_trace(model2, model2.defaults, "m/Lc", force, distance)
    """
    param_names = model.parameter_names
    if inverted_parameter not in params:
        raise ValueError(
            f"Inverted parameter {inverted_parameter} not in model parameters {params}."
        )

    for key in param_names:
        if key not in params:
            raise ValueError(f"Missing parameter {key} in supplied params.")

    # Grab reference parameter vector and index for the parameter list
    param_vector = [float(params[key]) for key in param_names]
    try:
        lb = params[inverted_parameter].lower_bound
        ub = params[inverted_parameter].upper_bound
    except AttributeError:
        raise ValueError(
            "The argument params takes a dictionary with `Parameter` values. This can be obtained "
            "from a fit by slicing it by a dataset (i.e. fit[dataset_handle]) or from a "
            "model (i.e. model.defaults). See help(parameter_trace) for more information."
        )
    inverted_parameter_index = param_names.index(inverted_parameter)

    def fit_single_point(x, y):
        x = np.asarray([x])

        def residual(inverted_parameter_value):
            param_vector[inverted_parameter_index] = inverted_parameter_value
            return y - model._raw_call(x, param_vector)

        def jacobian(inverted_parameter_value):
            param_vector[inverted_parameter_index] = inverted_parameter_value
            return -model.jacobian(x, param_vector)[inverted_parameter_index]

        jac = jacobian if model.has_jacobian else "2-point"
        result = scipy.optimize.least_squares(
            residual,
            param_vector[inverted_parameter_index],
            jac=jac,
            bounds=(lb, ub),
            method="trf",
            **kwargs,
        )

        return result.x[0]

    return np.asarray([fit_single_point(x, y) for (x, y) in zip(independent, dependent)])
