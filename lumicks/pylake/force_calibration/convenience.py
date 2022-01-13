import numpy as np
from lumicks.pylake.force_calibration.calibration_models import (
    ActiveCalibrationModel,
    PassiveCalibrationModel,
    FixedDiodeModel,
)
from lumicks.pylake.force_calibration.power_spectrum_calibration import (
    fit_power_spectrum,
    calculate_power_spectrum,
)


def calibrate_force(
    force_voltage_data,
    bead_diameter,
    temperature,
    *,
    viscosity=None,
    active_calibration=False,
    driving_data=np.asarray([]),
    driving_frequency_guess=37,
    axial=False,
    hydrodynamically_correct=False,
    rho_sample=None,
    rho_bead=1060.0,
    distance_to_surface=None,
    fast_sensor=False,
    sample_rate=78125,
    num_points_per_block=2000,
    fit_range=(1e2, 23e3),
    excluded_ranges=[],
    fixed_diode=None,
    drag=None,
):
    """Determine force calibration factors.

    The power spectrum calibration algorithm implemented here is based on [1]_, [2]_, [3]_, [4]_,
    [5]_, [6]_.

    References
    ----------
    .. [1] Berg-Sørensen, K. & Flyvbjerg, H. Power spectrum analysis for optical tweezers. Rev. Sci.
           Instrum. 75, 594 (2004).
    .. [2] Tolić-Nørrelykke, I. M., Berg-Sørensen, K. & Flyvbjerg, H. MatLab program for precision
           calibration of optical tweezers. Comput. Phys. Commun. 159, 225–240 (2004).
    .. [3] Hansen, P. M., Tolic-Nørrelykke, I. M., Flyvbjerg, H. & Berg-Sørensen, K.
           tweezercalib 2.1: Faster version of MatLab package for precise calibration of optical
           tweezers. Comput. Phys. Commun. 175, 572–573 (2006).
    .. [4] Berg-Sørensen, K., Peterman, E. J. G., Weber, T., Schmidt, C. F. & Flyvbjerg, H. Power
           spectrum analysis for optical tweezers. II: Laser wavelength dependence of parasitic
           filtering, and how to achieve high bandwidth. Rev. Sci. Instrum. 77, 063106 (2006).
    .. [5] Tolić-Nørrelykke, S. F, and Flyvbjerg, H, "Power spectrum analysis with least-squares
           fitting: amplitude bias and its elimination, with application to optical tweezers and
           atomic force microscope cantilevers." Review of Scientific Instruments 81.7 (2010)
    .. [6] Tolić-Nørrelykke S. F, Schäffer E, Howard J, Pavone F. S, Jülicher F and Flyvbjerg, H.
           Calibration of optical tweezers with positional detection in the back focal plane,
           Review of scientific instruments 77, 103101 (2006).

    Parameters
    ----------
    force_voltage_data : array_like
        Uncalibrated force data in volts.
    bead_diameter : float
        Bead diameter [um].
    temperature : float
        Liquid temperature [Celsius].
    viscosity : float, optional
        Liquid viscosity [Pa*s].
        When omitted, the temperature will be used to look up the viscosity of water at that
        particular temperature.
    active_calibration : bool, optional
        Active calibration, when set to True, driving_data must also be provided.
    driving_data : array_like, optional
        Array of driving data.
    driving_frequency_guess : float, optional
         Guess of the driving frequency.
    axial : bool, optional
        Is this an axial calibration? Only valid for a passive calibration.
    hydrodynamically_correct : bool, optional
        Enable hydrodynamically correct model.
    rho_sample : float, optional
        Density of the sample [kg/m**3]. Only used when using hydrodynamically correct model.
    rho_bead : float, optional
        Density of the bead [kg/m**3]. Only used when using hydrodynamically correct model.
    distance_to_surface : float, optional
        Distance from bead center to the surface [um]
        When specifying `None`, the model will use an approximation which is only suitable for
        measurements performed deep in bulk.
    fast_sensor : bool, optional
         Fast sensor? Fast sensors do not have the diode effect included in the model.
    sample_rate : float, optional
         Sample rate at which the signals were acquired.
    fit_range : tuple of float, optional
        Tuple of two floats (f_min, f_max), indicating the frequency range to use for the full model
        fit. [Hz]
    num_points_per_block : int, optional
        The spectrum is first block averaged by this number of points per block.
        Default: 2000.
    excluded_ranges : list of tuple of float, optional
        List of ranges to exclude specified as a list of (frequency_min, frequency_max).
    drag : float, optional
        Overrides the drag coefficient to this particular value.
    fixed_diode : float, optional
        Fix diode frequency to a particular frequency.
    """
    if active_calibration:
        if axial:
            raise ValueError("Active calibration is not supported for axial force.")
        if drag:
            raise ValueError("Drag coefficient cannot be carried over to active calibration.")

    if fixed_diode and fast_sensor:
        raise ValueError("When using fast_sensor=True, there is no diode model to fix.")

    if active_calibration and driving_data.size == 0:
        raise ValueError("Active calibration requires the driving_data to be defined.")

    model_params = {
        "bead_diameter": bead_diameter,
        "viscosity": viscosity,
        "temperature": temperature,
        "fast_sensor": fast_sensor,
        "distance_to_surface": distance_to_surface,
        "hydrodynamically_correct": hydrodynamically_correct,
        "rho_sample": rho_sample,
        "rho_bead": rho_bead,
    }

    model = (
        ActiveCalibrationModel(
            driving_data,
            force_voltage_data,
            sample_rate,
            driving_frequency_guess=driving_frequency_guess,
            **model_params,
        )
        if active_calibration
        else PassiveCalibrationModel(**model_params, axial=axial)
    )

    if drag:
        model._set_drag(drag)

    if fixed_diode:
        model._filter = FixedDiodeModel(fixed_diode)

    ps = calculate_power_spectrum(
        force_voltage_data,
        sample_rate,
        fit_range,
        num_points_per_block,
        excluded_ranges=excluded_ranges,
    )

    return fit_power_spectrum(ps, model)
