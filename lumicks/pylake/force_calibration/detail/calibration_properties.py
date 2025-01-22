import numpy as np
import tabulate

tabulate.PRESERVE_WHITESPACE = True


class CalibrationPropertiesMixin:
    def _get_parameter(self, pylake_key, bluelake_key):
        raise NotImplementedError

    @property
    def _parameters(self):
        """Properties that signify input parameters"""
        diode_params = (
            [
                "diode_relaxation_factor",
                "diode_frequency",
            ]
            if not self.fitted_diode
            else []
        )

        return [
            # Core calibration inputs
            "bead_diameter",
            "temperature",
            "viscosity",
            # Sensor parameters
            "fitted_diode",
            *diode_params,
            # Height calibration
            "distance_to_surface",
            # Active calibration parameters
            "driving_frequency_guess",
            "transferred_lateral_drag_coefficient",
            # Hydrodynamic parameters
            "hydrodynamically_correct",
            "rho_bead",
            "rho_sample",
            # Acquisition parameters
            "fit_range",
            "excluded_ranges",
            "sample_rate",
            "number_of_samples",
        ]  # omitted: fit_tolerance, max_iterations

    @property
    def _results(self):
        """Properties that signify results"""
        diode_results = (
            [
                "diode_relaxation_factor",
                "diode_frequency",
            ]
            if self.fitted_diode
            else []
        )

        return [
            # Core parameters
            "stiffness",
            "displacement_sensitivity",
            "force_sensitivity",
            "diffusion_constant",
            # Core parameters active
            "measured_drag_coefficient",
            # Fitting parameters
            "corner_frequency",
            "diffusion_constant_volts",
            *diode_results,
            # Active calibration diagnostics
            "driving_amplitude",
            "driving_frequency",
            "driving_power",
            # Theoretical drag coefficient
            "theoretical_bulk_drag",
            # Statistical diagnostics
            "backing",
            "chi_squared_per_degree",
            "stiffness_std_err",
            "displacement_sensitivity_std_err",
            "corner_frequency_std_err",
            "diffusion_volts_std_err",
            "diode_relaxation_factor_std_err",
            "diode_frequency_std_err",
            # Force offset
            "offset",
        ]

    def _print_properties(self, tablefmt="text"):
        def parse_prop(v):
            # Some of the fields are string-typed. As a result, the automatic floating point
            # formatting in tabulate fails which results in excessively long values.
            if np.isscalar(v):
                if isinstance(v, bool):
                    return " True" if v else "False"

                value = f"{v:.5g}"
                # Format value to 5 significant figures aligned at the decimal.
                return " " * (5 - (value + ".").find(".")) + value
            else:
                return str(v)

        return tabulate.tabulate(
            (
                (
                    prop,
                    getattr(CalibrationPropertiesMixin, prop).__doc__.split("\n")[0],
                    parse_prop(getattr(self, prop)),
                )
                for prop in self._results + self._parameters
                if getattr(self, prop) is not None
            ),
            tablefmt=tablefmt,
            headers=("Property", "Description", "Value"),
        )

    def _repr_html_(self):
        return self._print_properties(tablefmt="html")

    def __str__(self) -> str:
        return self._print_properties()

    @property
    def stiffness(self):
        """Trap stiffness (pN/nm)

        The trap stiffness gives a measure for how much force is required to displace the bead
        from the center of the optical trap. In passive calibration, this factor depends
        on a theoretical drag coefficient which is determined from the fluid viscosity, bead
        diameter and distance to the surface.

        In active calibration fewer assumptions are required to calculate the trap stiffness and
        the drag coefficient is inferred from the measurement data."""
        return self._get_parameter("kappa", "kappa (pN/nm)")

    @property
    def force_sensitivity(self):
        """Force sensitivity (pN/V)

        The force sensitivity provides the calibration factor for converting sensor readouts from
        voltages to forces. It is given by the product of the displacement sensitivity and
        trap stiffness.

        To recalibrate force channels, simply multiply them by the ratio of force sensitivities of
        the old and new calibration.
        """
        # We use the response in the Bluelake case, since it is always available. The force
        # sensitivity is not available for reset calibration to unity items for instance.
        return abs(self._get_parameter("Rf", "Response (pN/V)"))

    @property
    def displacement_sensitivity(self):
        """Displacement sensitivity (µm/V)

        The displacement sensitivity corresponds to the conversion factor from raw sensor voltages
        to the bead displacement from the trap center.

        In passive calibration, it is calculated from the ratio of the expected diffusion
        constant (in µm²/s) to the diffusion constant quantified from the power spectral fit
        (in V²/s). The former depends on the drag coefficient which in turn is linearly dependent
        on the viscosity and bead diameter, as well as non-linearly dependent on the distance
        from the flow-cell surface.

        In active calibration with a nano-stage, the displacement sensitivity is determined by
        applying a known displacement to the fluid in the flow-cell. Using parameters from the
        passive fit, we can calculate an expected motion of the bead in response to this fluid
        motion. The driving peak can be observed on the position sensitive detector (in Volts),
        while we know its motion in (microns). The advantage of active calibration is that it does
        not rely as strongly on a theoretical drag coefficient and is therefore less sensitivity to
        the input parameters viscosity, bead diameter and distance to the surface.

        To recalibrate distance channels, simply multiply them by the ratio of displacement
        sensitivities of the old and new calibration.
        """
        return self._get_parameter("Rd", "Rd (um/V)")

    @property
    def local_drag_coefficient(self):
        """Measured local drag coefficient (kg/s)

        .. note::

            This parameter is only available when using active calibration.
            Represents the drag at the height the calibration was performed. For the value corrected
            back to bulk, see :attr:`measured_drag_coefficient`.
        """
        return self._get_parameter("local_drag_coefficient", "local_drag_coefficient (kg/s)")

    @property
    def measured_drag_coefficient(self):
        """Measured bulk drag coefficient (kg/s)

        .. note::

            This parameter is only available when using active calibration.

            For surface calibrations where the distance to the surface was provided, this
            represents the bulk value as the model is used to calculate it back to what the
            drag would have been in bulk.
        """
        return self._get_parameter("gamma_ex", "gamma_ex (kg/s)")

    @property
    def corner_frequency(self):
        """Estimated corner frequency (Hz)

        .. note::

            When the hydrodynamically correct model is used and a height is provided, this returns
            the value in bulk. When this model is used, the height dependence due to the change in
            drag coefficient is captured by the model. In this case, any remaining
            height-dependent variation is due to optical aberrations.

            When using the simpler model or when not providing the height, the estimated corner
            frequency will depend on the distance to the surface resulting in a lower corner
            frequency near the surface.
        """
        return self._get_parameter("fc", "fc (Hz)")

    @property
    def theoretical_bulk_drag(self):
        r"""Expected bulk drag coefficient (kg/s)

        The expected drag coefficient in bulk for a spherical particle is given by:

        .. math::

            \gamma = 3 \pi \eta d

        Where :math:`d` represents the bead diameter, :math:`\eta` the liquid viscosity and
        :math:`T` the temperature.
        """
        return self._get_parameter("gamma_0", "gamma_0 (kg/s)")

    @property
    def diffusion_constant_volts(self):
        """Fitted diffusion constant (V²/s)

        .. note::

            When the hydrodynamically correct model is used and a height is provided, this returns
            the value in bulk. When this model is used, the height dependence due to the change in
            drag coefficient is captured by the model.

            When using the simpler model or when not providing the height, the estimated diffusion
            constant will depend on the distance to the surface resulting in a lower diffusion
            constant near the surface.
        """
        return self._get_parameter("D", "D (V^2/s)")

    @property
    def diffusion_constant(self):
        """Fitted diffusion constant (µm²/s)

        .. note::

            When the hydrodynamically correct model is used and a height is provided, this returns
            the value in bulk. When this model is used, the height dependence due to the change in
            drag coefficient is captured by the model.

            When using the simpler model or when not providing the height, the estimated diffusion
            constant will depend on the distance to the surface resulting in a lower diffusion
            constant near the surface.
        """
        if self.diffusion_constant_volts and self.displacement_sensitivity:
            return self.diffusion_constant_volts * self.displacement_sensitivity**2

    @property
    def diode_relaxation_factor(self):
        """Diode relaxation factor (-)

        The measured voltage (and thus the shape of the power spectrum) is determined by the
        Brownian motion of the bead within the trap as well as the response of the PSD to the
        incident light.

        For "fast" sensors, this second contribution is negligible at the frequencies typically
        fitted, while "standard" sensors exhibit a characteristic filtering where the PSD becomes
        less sensitive to changes in signal at high frequencies. This filtering effect is
        characterized by a constant that reflects the fraction of light that is transmitted
        instantaneously (the diode relaxation factor) and a corner frequency (diode_frequency).

        A relaxation factor of 1.0 results in no filtering. This property will return `None` for
        fast sensors (where the diode model is not taken into account).

        .. note::

            Some systems have characterized diodes. In these systems, this is not a fitted but
            fixed value. Please refer to the property `fitted_diode` to determine whether the diode
            frequency was fixed (pre-characterized) or fitted.
        """
        if alpha := self._get_parameter("alpha", "alpha"):
            return alpha  # Fitted diode

        return self._get_parameter("alpha", "Diode alpha")

    @property
    def diode_frequency(self):
        """Diode filtering frequency (Hz).

        The measured voltage (and thus the shape of the power spectrum) is determined by the
        Brownian motion of the bead within the trap as well as the response of the PSD to the
        incident light.

        For "fast" sensors, this second contribution is negligible at the frequencies typically
        fitted, while "standard" sensors exhibit a characteristic filtering where the PSD becomes
        less sensitive to changes in signal at high frequencies. This filtering effect is
        characterized by a constant that reflects the fraction of light that is transmitted
        instantaneously (the diode relaxation factor) and a corner frequency (diode_frequency).

        .. note::

            Some systems have characterized diodes. In these systems, this is not a fitted but
            fixed value. Please refer to the property `fitted_diode` to determine whether the diode
            frequency was fixed (pre-characterized) or fitted.
        """
        if f_diode := self._get_parameter("f_diode", "f_diode (Hz)"):
            return f_diode

        return self._get_parameter("f_diode", "Diode frequency (Hz)")  # Fixed diode

    @property
    def hydrodynamically_correct(self):
        """Hydrodynamically correct model.

        Force calibration involves fitting a model to the power spectrum of the calibration data.

        This power spectrum has a Lorentian shape when the viscous drag force is proportional
        to the bead velocity. This is true when the velocity field is stationary around the bead.
        This model is only appropriate at low frequencies or when using small beads.

        For larger beads, inertial effects start to play a role and the viscous force depends on
        the frequency of the motion and the second derivative of the bead position. These effects
        originate from waves generated by the interaction of the bead with the fluid, which in turn
        interact with the bead again. The hydrodynamically correct model takes these effects into
        account providing a more accurate description of the data especially for larger beads an
        at high frequencies."""
        return bool(
            self._get_parameter("Hydrodynamically correct", "Hydrodynamic correction enabled")
        )

    @property
    def fast_sensor(self):
        """Fast sensor

        Some force sensors include a parasitic filtering effect. When this flag is set to `False`,
        the model includes such a parasitic filtering effect. When this flag is `True`, no such
        effect is included in the model. This flag should only be used for fast sensors (sensors
        where the sensor does not exhibit a frequency dependent attenuation over the measured
        bandwidth)."""
        return not self.diode_frequency

    @property
    def bead_diameter(self):
        """Bead diameter (microns)"""
        return self._get_parameter("Bead diameter", "Bead diameter (um)")

    @property
    def viscosity(self):
        """Viscosity of the medium (Pa s)"""
        return self._get_parameter("Viscosity", "Viscosity (Pa*s)")

    @property
    def temperature(self):
        """Temperature (C)"""
        return self._get_parameter("Temperature", "Temperature (C)")

    @property
    def distance_to_surface(self):
        """Distance from bead center to surface (µm)"""
        return self._get_parameter("Distance to surface", "Bead center height (um)")

    @property
    def rho_sample(self):
        """Density of the medium (kg/m³).

        .. note::

            This parameter only affects hydrodynamically correct fits."""
        return self._get_parameter("Sample density", "Fluid density (Kg/m3)")

    @property
    def rho_bead(self):
        """Density of the bead (kg/m³).

        .. note::

            This parameter only affects hydrodynamically correct fits."""
        return self._get_parameter("Bead density", "Bead density (Kg/m3)")

    @property
    def backing(self):
        """Statistical backing (%)

        The support or backing is the probability that a repetition of the measurement that
        produced the data we fitted to will, after fitting, produce residuals whose squared sum is
        greater than the one we initially obtained. More informally, it represents the probability
        that a fit error at least this large should occur by chance.
        """
        return self._get_parameter("backing", "backing (%)")

    @property
    def stiffness_std_err(self):
        """Stiffness error (pN/nm)

        Obtained through Gaussian error propagation. See :attr:`stiffness` for more information."""
        # Note: Isn't exported by BL yet
        return self._get_parameter("err_kappa", "err_kappa (pN/nm)")

    @property
    def displacement_sensitivity_std_err(self):
        """Displacement sensitivity std error (µm/V)

        Obtained through Gaussian error propagation. See :attr:`displacement_sensitivity` for more
        information."""
        # Note: Isn't exported by BL yet
        return self._get_parameter("err_Rd", "err_Rd (um/V)")

    @property
    def corner_frequency_std_err(self):
        """Corner frequency std error (Hz)

        Asymptotic standard error of the fit. See :attr:`corner_frequency` for more information
        on this parameter."""
        return self._get_parameter("err_fc", "err_fc (Hz)")

    @property
    def diffusion_volts_std_err(self):
        """Diffusion constant std error (V²/s)

        Asymptotic standard error of the fit. See :attr:`diffusion_volts` for more information on
        this parameter."""
        return self._get_parameter("err_D", "err_D (V^2/s)")

    @property
    def diode_relaxation_factor_std_err(self):
        """Relaxation factor std error (-)

        Asymptotic standard error of the fit. See :attr:`diode_relaxation_factor` for more
        information on this parameter."""
        return self._get_parameter("err_alpha", "err_alpha")

    @property
    def diode_frequency_std_err(self):
        """Diode frequency std error (-)

        Asymptotic standard error of the fit. See :attr:`diode_frequency` for more information
        on this parameter."""
        return self._get_parameter("err_f_diode", "err_f_diode (Hz)")

    @property
    def chi_squared_per_degree(self):
        """Chi squared per degree of freedom

        This quantity is given by the sum of squares divided by the number of degrees of freedom
        and should be close to unity for a good fit."""
        return self._get_parameter("chi_squared_per_deg", "chi_squared_per_deg")

    @property
    def driving_frequency_guess(self):
        """Driving frequency guess (Hz)

        The force calibration procedure requires an initial guess of the frequency driving the
        active calibration procedure. This frequency is typically the requested driving frequency
        of the nano-stage. This initial estimate is refined using power spectral estimation.
        See :attr:`driving_frequency` for the achieved driving frequency.
        """
        return self._get_parameter("Driving frequency (guess)", "Driving data frequency (Hz)")

    @property
    def transferred_lateral_drag_coefficient(self):
        """Drag coefficient from lateral calibration

        When performing axial calibration, one can specify a known bulk drag coefficient obtained
        with active calibration and use that to calibrate the trap. This coefficient is then
        subsequently corrected for surface effects by applying Brenner's law and used in
        passive calibration."""
        return self._get_parameter("gamma_ex_lateral", "gamma_ex_lateral (kg/s)")

    @property
    def driving_amplitude(self):
        """Quantified driving amplitude (µm)

        This property corresponds to the driving amplitude quantified from the position signal
        measuring the stage or trap motion."""
        return self._get_parameter("driving_amplitude", "driving_amplitude (um)")

    @property
    def driving_frequency(self):
        """Quantified frequency (Hz)

        This property corresponds to the driving frequency quantified from the position signal
        measuring the stage or trap motion."""
        return self._get_parameter("driving_frequency", "driving_frequency (Hz)")

    @property
    def driving_power(self):
        """Driving power at the position sensitive detector (V²)

        This property corresponds to the driving power measured at the force detector (in Volts).
        The estimated :attr:`driving_frequency` is used to window the calibration data such that
        the driving peak ends up on exactly one frequency bin which can then subsequently be
        quantified. After quantification of the peak magnitude, the thermal background is
        subtracted to obtain the driving power.
        """
        # Note: Isn't exported by BL yet
        return self._get_parameter("driving_power", "driving_power (V^2)")

    @property
    def fitted_diode(self):
        """Diode parameters were fitted

        The measured voltage (and thus the shape of the power spectrum) is determined by the
        Brownian motion of the bead within the trap as well as the response of the PSD to the
        incident light.

        For "fast" sensors, this second contribution is negligible at the frequencies typically
        fitted, while "standard" sensors exhibit a characteristic filtering where the PSD becomes
        less sensitive to changes in signal at high frequencies. This filtering effect is
        characterized by a constant that reflects the fraction of light that is transmitted
        instantaneously (the diode relaxation factor) and a corner frequency (diode_frequency).

        Some systems have characterized diodes. In these systems, this is not a fitted but
        fixed value. This property is `False` for calibrations where the diode frequency and
        relaxation factor were fixed (pre-characterized)."""
        return self._fitted_diode

    @property
    def fit_range(self):
        """Spectral frequency range used for calibration (Hz)"""
        return self._fit_range

    @property
    def excluded_ranges(self):
        """Frequency exclusion ranges (Hz)

        When fitting power spectra, specific frequency intervals of the spectrum can be ignored to
        exclude noise peaks. This property returns a list of these intervals in Hertz."""
        return self._excluded_ranges

    @property
    def sample_rate(self):
        """Acquisition sample rate (Hz)."""
        raise NotImplementedError("This property is not defined for this item")

    @property
    def number_of_samples(self):
        """Number of fitted samples (-)."""
        raise NotImplementedError("This property is not defined for this item")

    @property
    def offset(self):
        """Force offset (pN)"""
        return self._get_parameter("NA", "Offset (pN)")

    @property
    def active_calibration(self):
        """Returns whether it was an active calibration or not

        Calibrations based on active calibration are less sensitive to assumptions about the
        bead diameter, viscosity, distance of the bead to the flow cell surface and temperature.
        During active calibration, the trap or nano-stage is oscillated sinusoidally. These
        oscillations result in a driving peak in the force spectrum. Using power spectral analysis,
        the force can then be calibrated without prior knowledge of the drag coefficient.

        While this results in improved accuracy, it may lead to an increase in the variability of
        results.

        .. note::

            When active calibration is performed using two beads, correction factors must be
            computed which account for the reduced flow field that forms around the beads due to
            the presence of a second bead.
        """
        return self.driving_frequency is not None

    @property
    def kind(self):
        kind = self._get_parameter("Kind", "Kind")
        if kind == "Full calibration":
            return "Active" if self.active_calibration else "Passive"
        else:
            return kind if kind is not None else "Unknown"
