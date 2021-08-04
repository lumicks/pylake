import numpy as np
import scipy.constants, scipy.signal


def calculate_active_term(time, driving_sinusoid, stiffness, gamma):
    """Simulate stage and bead position from simulation parameters.

    There's a term in the Langevin equation that can be used to take into account stage movement.
    Because the equations are linear, this term can be evaluated independently from the thermal
    part and added. See [1].

    The driving response is given by (Eq. 7 in [1]):

        x_response(t) = x_drive(t - t_lag) / sqrt(1 + (fc / fdrive)**2)

    With:

        fc = stiffness / (2.0 pi gamma)
        t_lag = (arctan(f_drive / fc) - pi/2) / (2 pi f_drive)

    [1] Tolić-Nørrelykke, S. F., Schäffer, E., Howard, J., Pavone, F. S., Jülicher, F., &
    Flyvbjerg, H. (2006). Calibration of optical tweezers with positional detection in the back
    focal plane. Review of scientific instruments, 77(10), 103101.

    Parameters
    ----------
    time : array_like
        Time axis [s]
    driving_sinusoid : tuple of float
        Amplitude [nm] and frequency [Hz] of active calibration stage movement
    stiffness : float
        Stiffness of the trap [N/m]
    gamma : float
        Friction coefficient [Ns/m]

    Returns
    -------
    stage_position : array_like
        Stage position in [m]
    bead_position_contribution : array_like
        Contribution to the bead position in [m]
    """
    amp_stage, f_stage = driving_sinusoid
    omega_stage = 2.0 * np.pi * f_stage
    stage_position = amp_stage * np.sin(omega_stage * time) * 1e-9

    fc = stiffness / (2.0 * np.pi * gamma)
    t_lag = (np.arctan(f_stage / fc) - np.pi / 2) / omega_stage
    amp_bead_position = amp_stage / np.sqrt(1.0 + (fc / f_stage) ** 2)
    bead_position_contribution = amp_bead_position * np.sin(omega_stage * (time - t_lag)) * 1e-9

    return stage_position, bead_position_contribution


def apply_diode_filter(positions, diode_alpha, diode_frequency, time_step):
    """Applies the simple diode filter model to the data.

    Parameters
    ----------
    positions : array_like
        List of bead positions [m]
    diode_alpha : float
        Fraction of the diode response that is instantaneous [-]
    diode_frequency : float
        Corner frequency of the filtering effect of the PSD [Hz]
    time_step : float
        Time step of the simulation [s]
    """
    assert 0 <= diode_alpha <= 1, (
        "Invalid diode fraction (alpha) provided. " "Should be between 0 and 1"
    )
    filter_coeff = np.exp(-2.0 * np.pi * time_step * diode_frequency)
    filtered = scipy.signal.lfilter([(1 - filter_coeff)], [1, -filter_coeff], positions)
    return positions * diode_alpha + filtered * (1.0 - diode_alpha)


def simulate_calibration_data(
    duration,
    sample_rate,
    bead_diameter,
    stiffness,
    viscosity,
    temperature,
    pos_response_um_volt,
    anti_aliasing,
    oversampling=8,
    driving_sinusoid=None,
    diode=None,
):
    """Simulate Brownian motion in optical trap

    This function simulates the Langevin equation that arises from a bead in an optical trap while
    the stage is oscillating. Rather than using a basic Euler scheme, we use the more exact
    simulation method from [1] which does not suffer from discretization error. Equation (72)
    from [1] reads:

        x[n+1] = c * x[n] + dx

    with:

        c = exp(-stiffness * dt / gamma)
        dx = sqrt( (1 - c^2) * diffusion * gamma / stiffness ) * normal(0, 1)

    For performance, we recast this simulation as a digital filter applied to a list of values drawn
    from a normal distribution rather than simulate it with a for loop.

    [1] Nørrelykke, S. F., & Flyvbjerg, H. (2011). Harmonic oscillator in heat bath: Exact
    simulation of time-lapse-recorded data and exact analytical benchmark statistics. Physical
    Review E, 83(4), 041103.

    Parameters
    ----------
    duration : float
        Time [s]
    sample_rate : int
        Sampling rate [Hz]
    bead_diameter : float
        Bead diameter [micron]
    stiffness : float
        Spring constant of the trap [pN/nm]
    viscosity : float
        Viscosity [Pa*s]
    temperature : float
        Temperature [C]
    pos_response_um_volt : float
        Response [um/V], also denoted in papers as Rd
    anti_aliasing : bool
        Should we anti-alias the data?
    oversampling : int
        Oversampling factor. Relatively high oversampling ratios are required to reject aliasing
        effectively.
    driving_sinusoid : tuple of float or None
        Parameters for the driving input.
        Amplitude [nm] and frequency of active [Hz] calibration stage movement
    diode : tuple of float or None
        Diode parameters:
        Alpha of the diode response that is instantaneous, ranges from 0 to 1 [-]
        Corner frequency of the filtering effect of the PSD [Hz]

    Returns
    -------
    time : np.ndarray
        Time in nanoseconds.
    volt : np.ndarray
        Position in volts.
    stage_pos : np.ndarray
        Stage position in microns.
    """
    boltzmann_const = scipy.constants.k
    gamma = 3.0 * np.pi * viscosity * bead_diameter * 1e-6  # friction coefficient [Ns/m]
    diffusion_const = (boltzmann_const * (temperature + 273.15)) / gamma  # diffusion [m^2/s]
    dt = 1 / sample_rate
    stiffness_si = stiffness * 1e-3  # Trap stiffness [N/m]

    # For stability, dt *must* be less than gamma / stiffness.
    dt_limit = gamma / stiffness_si
    if oversampling < int(np.ceil(dt / dt_limit)):
        raise RuntimeError("Oversampling ratio needs to be higher for stable simulation")

    oversampled_dt = dt / oversampling
    kappa_div_gamma_dt = (stiffness_si / gamma) * oversampled_dt
    c = np.exp(-kappa_div_gamma_dt)

    time = np.arange(0, duration, oversampled_dt)
    rand_scale = np.sqrt((gamma / stiffness_si) * diffusion_const * (1.0 - c * c))
    input_signal = rand_scale * np.random.standard_normal(time.shape)

    # Equation (72) from the paper can be rewritten as a digital filter with:
    #   b[0] = 1, a[0] = 1 and a[1] = -c.
    # One can see that from: a[0]*x[n] = b[0]*u[n] - a[1]*x[n-1].
    positions = scipy.signal.lfilter([1.0], [1.0, -c], input_signal)

    if diode:
        positions = apply_diode_filter(positions, *diode, oversampled_dt)

    # Decimate the signal
    positions = (
        scipy.signal.decimate(positions, oversampling, ftype="fir")
        if anti_aliasing
        else positions[::oversampling]
    )
    time = time[::oversampling]

    if driving_sinusoid:
        # Because the SD equations are linear, this term can be evaluated independently.
        stage_m, dx_active_m = calculate_active_term(time, driving_sinusoid, stiffness_si, gamma)
        positions += dx_active_m
    else:
        stage_m = np.zeros(time.shape)

    # Convert from meters to voltage
    positions_volt = positions * 1e6 / pos_response_um_volt

    return positions_volt, stage_m * 1e6
