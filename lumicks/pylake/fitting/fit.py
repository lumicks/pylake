from collections import OrderedDict

import numpy as np
import scipy

from .model import Model
from .datasets import Datasets, FdDatasets
from .parameters import Params
from .detail.utilities import print_styled, optimal_plot_layout
from ..detail.utilities import unique, lighten_color
from .profile_likelihood import ProfileLikelihood1D
from .detail.derivative_manipulation import numerical_jacobian


def front(x):
    return next(iter(x))


class Fit:
    """Object which is used for fitting. It is a collection of models and their data. Once data is
    loaded, a fit object contains parameters, which can be fitted by invoking fit.

    Parameters
    ----------
    *models
        Variable number of `pylake.fitting.Model`.

    Examples
    --------
    ::

        from lumicks import pylake

        dna_model = pylake.ewlc_odijk_force("DNA")
        fit = pylake.FdFit(dna_model)
        data = fit.add_data("Dataset 1", force, distance)

        fit["DNA/Lp"].lower_bound = 35  # Set lower bound for DNA Lp
        fit["DNA/Lp"].upper_bound = 80  # Set upper bound for DNA Lp
        fit.fit()

        fit.plot("Dataset 1", "k--")  # Plot the fitted model
    """

    def __init__(self, *models):
        self.models = {m.uuid: m for m in models}
        self.datasets = {m.uuid: self._dataset(m) for m in models}
        self._params = Params()
        self._invalidate_build()

    def _dataset(self, model):
        return Datasets(model, self)

    def update_params(self, other):
        """Sets parameters if they are found in the target fit.

        Parameters
        ----------
        other : Fit or Params
        """
        if isinstance(other, Params):
            self.params.update_params(other)
        elif isinstance(other, self.__class__):
            self.params.update_params(other.params)
        else:
            raise RuntimeError("Did not pass compatible argument to update_params")

    def __getitem__(self, item):
        if isinstance(item, Model):
            return self.datasets[item.uuid]
        elif len(self.datasets) == 1 and item in front(self.datasets.values()).names:
            return self.params[front(self.datasets.values()).data[item]]
        else:
            return self.params[item]

    @property
    def data(self):
        if len(self.datasets) > 1:
            raise RuntimeError(
                "This Fit is comprised of multiple models. Please access data for a particular "
                "model by invoking fit[model].data[dataset_name]."
            )

        return front(self.datasets.values()).data

    def _add_data(self, name, x, y, params=None):
        if len(self.datasets) > 1:
            raise RuntimeError(
                "This Fit is comprised of multiple models. Please add data to a particular model "
                "by invoking fit[model].add_data(...)"
            )

        return front(self.datasets.values())._add_data(name, x, y, params)

    @property
    def has_jacobian(self):
        """Returns true if it is possible to evaluate the Jacobian of the fit."""
        has_jacobian = True
        for model in self.models.values():
            has_jacobian = has_jacobian and model.has_jacobian

        return has_jacobian

    @property
    def n_residuals(self):
        """Number of data points."""
        self._rebuild()
        count = 0
        for data in self.datasets.values():
            count += data.n_residuals

        return count

    @property
    def n_params(self):
        """Number of parameters in the Fit"""
        self._rebuild()
        return len(self._params)

    @property
    def params(self):
        """Fit parameters. See also `pylake.fitting.parameters.Params`"""
        self._rebuild()
        return self._params

    @property
    def dirty(self):
        """Validate that all the Datasets that we are about the fit were actually linked."""
        dirty = not self._built
        for data in self.datasets.values():
            dirty = dirty or not data.built

        return dirty

    def _rebuild(self):
        """Checks whether the model state is up to date. Any user facing methods should ideally
        check whether the model needs to be rebuilt."""
        if self.dirty:
            self._build_fit()

    def _invalidate_build(self):
        self._built = False

    def _build_fit(self):
        """This function generates the global parameter list from the parameters of the individual
        sub models. It also generates unique conditions from the data specification."""
        all_parameter_names = [
            p for data_set in self.datasets.values() for p in data_set._transformed_params
        ]
        all_defaults = [d for data_set in self.datasets.values() for d in data_set._defaults]
        unique_parameter_names = unique(all_parameter_names)
        parameter_lookup = OrderedDict(
            zip(unique_parameter_names, np.arange(len(unique_parameter_names)))
        )

        for data in self.datasets.values():
            data._link_data(parameter_lookup)

        defaults = [all_defaults[all_parameter_names.index(n)] for n in unique_parameter_names]
        self._params._set_params(unique_parameter_names, defaults)
        self._built = True

    def _prepare_fit(self):
        """Checks whether the model is ready for fitting and returns the current parameter values, which parameters are
        fitted and the parameter bounds."""
        self._rebuild()
        if not self.n_residuals:
            raise RuntimeError("This model has no data associated with it.")

        if not np.any(self.params.fitted):
            raise RuntimeError("This model has no free parameters. There is nothing to fit.")

        return (
            self.params.values,
            self.params.fitted,
            self.params.lower_bounds,
            self.params.upper_bounds,
        )

    def _fit(self, parameter_vector, lb, ub, fitted, show_fit=False, **kwargs):
        """Fit the model

        Parameters
        ----------
        parameter_vector : array_like
            List of parameters
        lb : array_like
            list of lower parameter bounds
        ub : array_like
            list of lower parameter bounds
        show_fit : bool
            show fitting (slow!)
        fitted : array_like
            list of which parameters are fitted
        """
        import matplotlib.pyplot as plt

        if show_fit:
            fig = plt.gcf()

        def residual(params):
            parameter_vector[fitted] = params

            if show_fit:
                parameter_names = self.params.keys()
                for name, value in zip(parameter_names, parameter_vector):
                    self.params[name] = value
                plt.figure(fig.number)
                for model in self.models.values():
                    self[model].plot()
                fig.canvas.draw()
                fig.canvas.flush_events()
                fig.clf()

            return self._calculate_residual(parameter_vector)

        def jacobian(params):
            parameter_vector[fitted] = params
            return self._calculate_jacobian(parameter_vector)[:, fitted]

        result = scipy.optimize.least_squares(
            residual,
            parameter_vector[fitted],
            jac=jacobian if self.has_jacobian else "2-point",
            bounds=(lb[fitted], ub[fitted]),
            method="trf",
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
            **kwargs,
        )

        parameter_vector[fitted] = result.x

        return parameter_vector

    def fit(self, show_fit=False, **kwargs):
        """Fit the model

        Parameters
        ----------
        show_fit : bool
            Show the fitting procedure as it is progressing.

        Raises
        ------
        ValueError
            If the initial parameters are outside the parameter bounds.
        RuntimeError
            If this `Fit` has no data associated with it.
        RuntimeError
            If this `Fit` has no free parameters.
        """
        parameter_vector, fitted, lb, ub = self._prepare_fit()

        out_of_bounds = np.logical_or(
            parameter_vector[fitted] < lb[fitted], parameter_vector[fitted] > ub[fitted]
        )
        if np.any(out_of_bounds):
            raise ValueError(
                f"Initial parameters {self.params.keys()[fitted][out_of_bounds]} are outside the "
                "parameter bounds. Please set the model parameters, lower_bound and upper_bound "
                "for these parameters to consistent values."
            )

        parameter_vector = self._fit(parameter_vector, lb, ub, fitted, show_fit=show_fit, **kwargs)

        parameter_names = self.params.keys()
        for name, value in zip(parameter_names, parameter_vector):
            self.params[name] = value

        if self.has_jacobian:
            for name, value in zip(parameter_names[fitted], np.diag(self.cov)):
                self.params[name].stderr = np.sqrt(value)

            for name in parameter_names[np.logical_not(fitted)]:
                self.params[name].stderr = None

        return self

    def profile_likelihood(
        self,
        parameter_name,
        min_step=1e-4,
        max_step=1.0,
        num_steps=100,
        step_factor=2.0,
        min_chi2_step=0.05,
        max_chi2_step=0.25,
        termination_significance=0.99,
        confidence_level=0.95,
        verbose=False,
    ):
        """Calculate a profile likelihood. This method traces an optimal path through parameter
        space in order to estimate parameter confidence intervals. It iteratively performs a step
        for the profiled parameter, then fixes that parameter and re-optimizes all the other
        parameters [4]_ [5]_.

        Parameters
        ----------
        parameter_name: str
            Which parameter to evaluate a profile likelihood for.
        min_step: float
            Minimum step size. This is multiplied by the current parameter value to come to a
            minimum step size used in the step-size estimation procedure.
        max_step: float
            Maximum step size.
        num_steps: integer
            Number of steps to take .
        step_factor: float
            Which factor to change the step-size by when step-size is too large or too small.
        min_chi2_step: float
            Minimal desired step in terms of chi squared change prior to re-optimization. When the
            step results in a fit change smaller than this threshold, the step-size will be
            increased.
        max_chi2_step: float
            Minimal desired step in terms of chi squared change prior to re-optimization. When the
            step results in a fit change bigger than this threshold, the step-size will be reduced.
        termination_significance: float
            Significance level for terminating the parameter scan. When the fit quality exceeds the
            termination_significance confidence level, it stops scanning.
        confidence_level: float
            Significance level for the chi squared test.
        verbose: bool
            Controls the verbosity of the output.

        Raises
        ------
        KeyError
            If `parameter_name` is not present in the `Fit`.
        RuntimeError
            If `parameter_name` is a fixed parameter in the `Fit`.
        ValueError
            If `max_step` < `min_step`.
        ValueError
            If `max_chi2_step` < `min_chi2_step`.

        References
        ----------
        .. [4] Raue, A., Kreutz, C., Maiwald, T., Bachmann, J., Schilling, M., Klingmüller, U.,
               & Timmer, J. (2009). Structural and practical identifiability analysis of partially
               observed dynamical models by exploiting the profile likelihood. Bioinformatics,
               25(15), 1923-1929.
        .. [5] Maiwald, T., Hass, H., Steiert, B., Vanlier, J., Engesser, R., Raue, A., Kipkeew,
               F., Bock, H.H., Kaschek, D., Kreutz, C. and Timmer, J., 2016. Driving the model to
               its limit: profile likelihood based model reduction. PloS one, 11(9).
        """

        if parameter_name not in self.params:
            raise KeyError(f"Parameter {parameter_name} not present in fitting object.")

        if self.params[parameter_name].fixed:
            raise RuntimeError(f"Parameter {parameter_name} is fixed in the fitting object.")

        if max_step <= min_step:
            raise ValueError(
                f"max_step must be larger than min_step, got max_step={max_step} and "
                f"min_step={min_step}"
            )

        if max_chi2_step <= min_chi2_step:
            raise ValueError(
                "max_chi2_step must be larger than min_chi2_step, got "
                f"max_chi2_step={max_chi2_step} and min_chi2_step={min_chi2_step}"
            )

        profile = ProfileLikelihood1D(
            parameter_name,
            min_step,
            max_step,
            step_factor,
            min_chi2_step,
            max_chi2_step,
            termination_significance,
            confidence_level,
            1,
        )

        def trial(parameters=None):
            return -2.0 * self.log_likelihood(parameters, self.sigma)

        profile._extend_profile(trial, self._fit, self.params, num_steps, True, verbose)
        profile._extend_profile(trial, self._fit, self.params, num_steps, False, verbose)

        self.params[parameter_name].profile = profile

        return profile

    def _calculate_residual(self, parameter_values=None):
        self._rebuild()
        if parameter_values is None:
            parameter_values = self.params.values

        residual_idx = 0
        residual = np.zeros(self.n_residuals)
        for model in self.models.values():
            current_residual = model._calculate_residual(
                self.datasets[model.uuid], parameter_values
            )
            current_n = len(current_residual)
            residual[residual_idx : residual_idx + current_n] = current_residual
            residual_idx += current_n

        return residual

    def _calculate_jacobian(self, parameter_values=None):
        self._rebuild()
        if parameter_values is None:
            parameter_values = self.params.values

        residual_idx = 0
        jacobian = np.zeros((self.n_residuals, len(parameter_values)))
        for model in self.models.values():
            current_jacobian = model._calculate_jacobian(
                self.datasets[model.uuid], parameter_values
            )
            current_n = current_jacobian.shape[0]
            jacobian[residual_idx : residual_idx + current_n, :] = current_jacobian
            residual_idx += current_n

        return jacobian

    def verify_jacobian(self, params, plot=0, verbose=True, dx=1e-6, **kwargs):
        self._rebuild()
        if len(params) != len(self._params):
            raise ValueError(
                "Parameter vector has invalid length. "
                f"Expected: {len(self._params)}, got: {len(params)}."
            )

        jacobian = self._calculate_jacobian(params).transpose()
        jacobian_fd = numerical_jacobian(self._calculate_residual, params, dx)

        if plot:
            import matplotlib.pyplot as plt

            n_x, n_y = optimal_plot_layout(len(self.params))
            for i_parameter, parameter in enumerate(self.params):
                plt.subplot(n_x, n_y, i_parameter + 1)
                plt.plot(np.transpose(jacobian[i_parameter, :]), linewidth=2)
                plt.plot(np.transpose(jacobian_fd[i_parameter, :]), "--", linewidth=1)
                plt.title(parameter)
                plt.legend(["Analytic", "FD"])

        is_close = np.allclose(jacobian, jacobian_fd, **kwargs)
        if not is_close:
            parameter_names = list(self.params.keys())
            if verbose:
                maxima = np.max(jacobian - jacobian_fd, axis=1)
                for i, v in enumerate(maxima):
                    if np.allclose(jacobian[i, :], jacobian_fd[i, :]):
                        print(f"Parameter {parameter_names[i]}({i}): {v}")
                    else:
                        print_styled("warning", f"Parameter {parameter_names[i]}({i}): {v}")

        return is_close

    def plot(
        self,
        data=None,
        fmt="",
        independent=None,
        legend=True,
        plot_data=True,
        overrides=None,
        **kwargs,
    ):
        """Plot model and data

        Parameters
        ----------
        data : str
            Name of the data set to plot (optional, omission plots all for that model).
        fmt : str
            Format string, forwarded to :func:`matplotlib.pyplot.plot`.
        independent : array_like
            Array with values for the independent variable (used when plotting the model).
        legend : bool
            Show legend.
        plot_data : bool
            Show data.
        overrides : dict
            Parameter / value pairs which override parameter values in the current fit. Should be
            a dict of ``{str: float}`` that provides values for parameters which should be set to
            particular values in the plot.
        ``**kwargs``
            Forwarded to :func:`matplotlib.pyplot.plot`.

        Raises
        ------
        KeyError
            If the argument `data` is supplied but no dataset with that name is found in the `Fit`.
        RuntimeError
            If the `Fit` uses multiple models, but no `Model` is selected beforehand.

        Examples
        --------
        ::

            from lumicks import pylake

            model = pylake.ewlc_odijk_force("DNA")
            fit = pylake.FdFit(model)
            fit.add_data("Control", force, distance)
            fit.fit()

            # Basic plotting of one data set over a custom range can be done by just invoking plot.
            fit.plot("Control", 'k--', np.arange(2.0, 5.0, 0.01))

            # Have a quick look at what a stiffness of 5 would do to the fit.
            fit.plot("Control", overrides={"DNA/St": 5})

            # When dealing with multiple models in one fit, one has to select the model first when
            # we want to plot.
            model1 = pylake.ewlc_odijk_distance("DNA")
            model2 = pylake.ewlc_odijk_distance("DNA") + pylake.ewlc_odijk_distance("protein")
            fit[model1].add_data("Control", force1, distance2)
            fit[model2].add_data("Control", force1, distance2)
            fit.fit()

            fit = pylake.FdFit(model1, model2)
            fit[model1].plot("Control")  # Plots data set Control for model 1
            fit[model2].plot("Control")  # Plots data set Control for model 2
        """
        if len(self.models) > 1:
            raise RuntimeError("Please select a model to plot using fit[model].plot(...).")

        self._plot(
            front(self.models.values()),
            data,
            fmt,
            overrides,
            independent,
            legend,
            plot_data,
            **kwargs,
        )

    def _plot(self, model, data, fmt, overrides, independent, legend, plot_data, **kwargs):
        import matplotlib.pyplot as plt

        self._rebuild()

        params, _ = self._override_params(overrides)
        dataset = self.datasets[model.uuid]

        def plot(fit_data):
            x_values = fit_data.x if independent is None else independent
            label = kwargs.pop("label", fit_data.name)
            model_lines = model.plot(
                params[fit_data],
                x_values,
                fmt,
                **kwargs,
                zorder=1,
                label=label + " (model)",
            )

            if plot_data:
                color = model_lines[0].get_color()
                fit_data.plot(
                    ".",
                    **kwargs,
                    color=lighten_color(color, -0.3),
                    zorder=0,
                    label=label + " (data)",
                )

        if data:
            if data not in dataset.data:
                raise KeyError(f"Did not find dataset with name {data}")
            plot(dataset.data[data])
        else:
            for data in dataset.data.values():
                plot(data)

        if legend:
            plt.legend()

    def _override_params(self, overrides=None):
        from copy import deepcopy

        params = self.params

        params = deepcopy(params)
        if overrides:
            for key, value in overrides.items():
                if key in params:
                    params[key] = value
                else:
                    raise KeyError(f"Parameter {key} is not a parameter used in the fit")

        return params, overrides

    @property
    def sigma(self):
        """Error variance of the data points."""
        # TO DO: Ideally, this will eventually depend on the exact error model used. For now,
        # we use the a-posteriori noise standard deviation estimate based on the residual.
        res = self._calculate_residual()
        est_sd = np.sqrt((res * res).sum() / (len(res) - np.sum(self.params.fitted)))

        # This is toy data. Computing a variance estimate here makes no sense.
        if est_sd == 0:
            return np.ones(len(res))

        return est_sd * np.ones(len(res))

    def log_likelihood(self, params=None, sigma=None):
        """The model residual is given by chi squared = -2 log(L)"""
        self._rebuild()
        res = self._calculate_residual(params)
        sigma = sigma if np.any(sigma) else self.sigma
        weighted_sum = -np.sum((res / sigma) ** 2) / 2.0
        if np.isnan(weighted_sum):
            raise RuntimeError(
                "Residual returned NaN. Model cannot be evaluated at these parameter values."
            )

        return (
            -(self.n_residuals / 2.0) * np.log(2.0 * np.pi) - np.sum(np.log(sigma)) + weighted_sum
        )

    @property
    def aic(self):
        r"""Calculates the Akaike Information Criterion:

        .. math::
            AIC = 2 k - 2 \ln{(L)}

        Where k refers to the number of parameters, n to the number of observations (or data
        points) and L to the maximized value of the likelihood function [6]_.

        The emphasis of this criterion is future prediction. It does not lead to consistent model
        selection and is more prone to over-fitting than the Bayesian Information Criterion.

        References
        ----------
        .. [6] Cavanaugh, J.E., 1997. Unifying the derivations for the Akaike and corrected Akaike
               information criteria. Statistics & Probability Letters, 33(2), pp.201-208.
        """
        self._rebuild()
        k = sum(self.params.fitted)
        LL = self.log_likelihood()
        return 2.0 * k - 2.0 * LL

    @property
    def aicc(self):
        r"""Calculates the Corrected Akaike Information Criterion:

        .. math::
            AICc = AIC + \frac{2 k^2 + 2 k}{n - k - 1}

        Where k refers to the number of parameters, n to the number of observations (or data
        points) and L to the maximized value of the likelihood function [7]_.

        The emphasis of this criterion is future prediction. Compared to the AIC it should be less
        prone to overfitting for smaller sample sizes. Analogously to the AIC, it does not lead to
        a consistent model selection procedure.

        References
        ----------
        .. [7] Cavanaugh, J.E., 1997. Unifying the derivations for the Akaike and corrected Akaike
               information criteria. Statistics & Probability Letters, 33(2), pp.201-208.
        """

        aic = self.aic
        k = sum(self.params.fitted)
        return aic + (2.0 * k * k + 2.0 * k) / (self.n_residuals - k - 1.0)

    @property
    def bic(self):
        r"""Calculates the Bayesian Information Criterion:

        .. math::
            BIC = k \ln{(n)} - 2 \ln{(L)}

        Where k refers to the number of parameters, n to the number of observations (or data points)
        and L to the maximized value of the likelihood function

        The emphasis of the BIC is put on parsimonious models. As such it is less prone to
        over-fitting. Selection via BIC leads to a consistent model selection procedure, meaning
        that as the number of data points tends to infinity, BIC will select the true model
        assuming the true model is in the set of selected models.
        """

        k = sum(self.params.fitted)
        return k * np.log(self.n_residuals) - 2.0 * self.log_likelihood()

    @property
    def cov(self):
        """Returns the inverse of the approximate Hessian. This approximation is valid when the
        model fits well (small residuals) and there is sufficient data to assume we're in the
        asymptotic regime.

        It makes use of the Gauss-Newton approximation of the Hessian, which uses only the first
        order sensitivity information. This is valid for linear problems and problems near the
        optimum (assuming the model fits) [8]_ [9]_.

        References
        ----------
        .. [8] Press, W.H., Teukolsky, S.A., Vetterling, W.T. and Flannery, B.P., 1988. Numerical
               recipes in C.
        .. [9] Maiwald, T., Hass, H., Steiert, B., Vanlier, J., Engesser, R., Raue, A., Kipkeew,
               F., Bock, H.H., Kaschek, D., Kreutz, C. and Timmer, J., 2016. Driving the model to
               its limit: profile likelihood based model reduction. PloS one, 11(9).
        """
        # Note that this approximation is only valid if the noise on each data set is the same.
        if self.has_jacobian:
            J = self._calculate_jacobian()[:, self.params.fitted]
            J = J / np.transpose(np.tile(self.sigma, (J.shape[1], 1)))
            return np.linalg.pinv(np.transpose(J).dot(J))
        else:
            raise NotImplementedError(
                "In order to calculate a covariance matrix, a model Jacobian has to be specified "
                "for the model."
            )

    def _repr_html_(self):
        out_string = "<h4>Fit</h4>\n"

        for model in self.models.values():
            datasets = "".join(f"{self.datasets[model.uuid]._repr_html_()}<br>\n")
            out_string += f"<h5>Model: {model.name}</h5>\n"
            eqn = model.get_formatted_equation_string(tex=True)
            if eqn:
                out_string += f"<h5>&ensp;Equation:</h5>${eqn}$<br>\n"
            out_string += f"<h5>&ensp;Data:</h5>\n{datasets}<br>"

        return out_string + f"<h5>&ensp;Fitted parameters:</h5>\n{self.params._repr_html_()}"

    def __repr__(self):
        return (
            f"lumicks.pylake.{self.__class__.__name__}"
            f"(models={{{', '.join([x.name for x in self.models.values()])}}}, "
            f"N={self.n_residuals})"
        )

    def __str__(self):
        indent = 2
        out_string = "Fit\n"

        for model in self.models.values():
            datasets = (" " * indent + "- " + self.datasets[model.uuid].__str__()).splitlines(True)
            datasets = (" " * (2 * indent)).join(datasets)

            out_string += f"{' ' * indent}- Model: {model.name}\n"

            eqn = model.get_formatted_equation_string(tex=False)
            if eqn:
                out_string += f"{' ' * indent}- Equation:\n      {eqn}\n\n{datasets}"

        return out_string + (
            f"\n{' ' * indent}- Fitted parameters:\n"
            f"{(' ' * (2 * indent))}"
            f"{(' ' * (2 * indent)).join(self.params.__str__().splitlines(True))}"
        )


class FdFit(Fit):
    """Object which is used for fitting. It is a collection of models and their data. Once data is
    loaded, a fit object contains parameters, which can be fitted by invoking fit.

    Examples
    --------
    ::

        from lumicks import pylake

        dna_model = pylake.ewlc_odijk_force("DNA")
        fit = pylake.FdFit(dna_model)
        data = fit.add_data("Dataset 1", force, distance)

        fit["DNA/Lp"].lower_bound = 35  # Set lower bound for DNA Lp
        fit["DNA/Lp"].upper_bound = 80  # Set upper bound for DNA Lp
        fit.fit()

        fit.plot("Dataset 1", "k--")  # Plot the fitted model"""

    def add_data(self, name, f, d, params=None):
        """Adds a data set to this fit.

        Parameters
        ----------
        name : str
            Name of this data set.
        f : array_like
            An array_like containing force data.
        d : array_like
            An array_like containing distance data.
        params : Optional[dict of {str : str or int}]
            List of parameter transformations. These can be used to convert one parameter in the
            model, to a new parameter name or constant for this specific data set (for more
            information, see the examples).

        Examples
        --------
        ::

            dna_model = pylake.ewlc_odijk_force("DNA")  # Use an inverted Odijk eWLC model.
            fit = pylake.FdFit(dna_model)

            fit.add_data("Data1", force1, distance1)  # Load the first data set like that
            fit.add_data("Data2", force2, distance2, params={"DNA/Lc": "DNA/Lc_RecA"})  # Different DNA/Lc
        """
        if front(self.models.values()).independent == "f":
            return self._add_data(name, f, d, params)
        else:
            return self._add_data(name, d, f, params)

    def _dataset(self, model):
        return FdDatasets(model, self)
