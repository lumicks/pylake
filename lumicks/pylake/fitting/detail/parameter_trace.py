from .utilities import parse_transformation
import numpy as np
import scipy as sp
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
    **kwargs
        parameter renames (e.g. protein_Lc="protein_Lc_1")

    Examples
    --------
    ::
        # Define the model to be fitted
        M_protein = force_model("protein", "invWLC") + force_model("f", "offset")

        # Fit the overall model first
        M_protein.load_data(distances_corrected, forces)
        protein_fit = FitObject(M_protein)
        protein_fit.fit()

        # Calculate a per data point contour length
        lcs = parameter_trace(M_protein, protein_fit.parameters, "protein_Lc", distances, forces)
    """
    parameter_names = list(parse_transformation(model.parameter_names, **kwargs).keys())
    assert inverted_parameter in parameters, f"Inverted parameter not in model parameter vector {parameters}."
    for key in parameter_names:
        assert key in parameters, f"Missing parameter {key} in supplied parameter vector."

    # Grab reference parameter vector and index for the parameter list
    parameter_vector = [parameters[key].value for key in parameter_names]
    lb = parameters[inverted_parameter].lb
    ub = parameters[inverted_parameter].ub
    inverted_parameter_index = parameter_names.index(inverted_parameter)

    def residual(inverted_parameter_values):
        parameter_vector[inverted_parameter_index] = inverted_parameter_values
        return dependent - model._raw_call(independent, parameter_vector)

    def jacobian(inverted_parameter_values):
        parameter_vector[inverted_parameter_index] = inverted_parameter_values
        return -sp.sparse.diags(model.jacobian(independent, parameter_vector)[inverted_parameter_index, :], offsets=0)

    initial_estimate = np.ones(independent.shape) * parameter_vector[inverted_parameter_index]

    jac = jacobian if model.has_jacobian else "2-point"
    result = optim.least_squares(residual, initial_estimate, jac=jac,
                                 jac_sparsity=sp.sparse.identity(len(independent)),
                                 bounds=(lb, ub), method='trf', ftol=1e-08, xtol=1e-08, gtol=1e-8)

    return result.x
