import os

import numpy as np
import pytest

import lumicks.pylake.force_calibration.power_spectrum_calibration as psc
from lumicks.pylake.low_level import make_continuous_slice
from lumicks.pylake.detail.imaging_mixins import _FIRST_TIMESTAMP
from lumicks.pylake.force_calibration.calibration_models import PassiveCalibrationModel

from .test_calibration_item import ref_active
from .data.simulate_calibration_data import generate_active_calibration_test_data


class ReferenceModels:
    @staticmethod
    def lorentzian(f, fc, diffusion_constant):
        # Lorentzian in V^2/Hz
        return diffusion_constant / (np.pi**2) / (fc**2 + f**2)

    @staticmethod
    def lorentzian_filtered(f, fc, diffusion_constant, alpha, f_diode):
        return ReferenceModels.lorentzian(f, fc, diffusion_constant) * (
            alpha**2 + (1.0 - alpha**2) / (1.0 + (f / f_diode) ** 2)
        )

    @staticmethod
    def lorentzian_td(corner_frequency, diffusion_constant, alpha, f_diode, num_samples):
        f = np.arange(0, num_samples)
        power_spectrum = ReferenceModels.lorentzian_filtered(
            f, corner_frequency, diffusion_constant, alpha, f_diode
        )
        data = np.fft.irfft(np.sqrt(np.abs(power_spectrum * 0.5)), norm="forward")
        return data, len(data)


@pytest.fixture
def reference_models():
    return ReferenceModels()


@pytest.fixture
def integration_test_parameters():
    return {
        "bead_diameter": 1.03,
        "viscosity": 1.1e-3,
        "temperature": 25,
        "hydrodynamically_correct": True,
        "rho_sample": 997.0,
        "rho_bead": 1040.0,
        "distance_to_surface": 1.51 * 1.03 / 2,
    }, {
        "sample_rate": 78125,
        "stiffness": 0.1,
        "pos_response_um_volt": 0.618,
        "driving_sinusoid": (500, 31.95633),
        "diode": (0.4, 15000),
    }


@pytest.fixture
def mack_parameters():
    return {
        "wavelength_nm": 1064.0,
        "refractive_index_medium": 1.333,
        "surface_position": 101.65754300557573,
        "displacement_sensitivity": 9.724913160609043,
        "intensity_amplitude": -0.10858224326787835,
        "intensity_phase_shift": 1.6535670092299886,
        "intensity_decay_length": 0.308551871490813,
        "scattering_polynomial_coeffs": [
            -0.043577454353825644,
            0.22333743993836863,
            -0.33331150250090585,
            0.1035148152731559,
        ],
        "focal_shift": 0.921283446497108,
        "nonlinear_shift": 0.0,
    }


@pytest.fixture(scope="session")
def active_ref_data():
    np.random.seed(37738006)

    volts, stage = generate_active_calibration_test_data(
        duration=10,
        bead_diameter=ref_active["Bead diameter (um)"],
        viscosity=ref_active["Viscosity (Pa*s)"],
        temperature=ref_active["Temperature (C)"],
        stiffness=ref_active["kappa (pN/nm)"],
        pos_response_um_volt=ref_active["Rd (um/V)"],
        driving_sinusoid=(
            ref_active["driving_amplitude (um)"] * 1e3,  # um -> nm
            ref_active["driving_frequency (Hz)"],
        ),
        diode=(ref_active["alpha"], ref_active["f_diode (Hz)"]),
        hydrodynamically_correct=ref_active["Hydrodynamic correction enabled"],
        rho_sample=ref_active["Fluid density (Kg/m3)"],
        rho_bead=ref_active["Bead density (Kg/m3)"],
        distance_to_surface=ref_active.get("Bead center height (um)"),
        sample_rate=int(ref_active["Sample rate (Hz)"]),
    )

    def make_slice(data):
        return make_continuous_slice(data, _FIRST_TIMESTAMP + int(1e9), int(1e9 / 78125))

    return make_slice(volts), make_slice(stage)


@pytest.fixture(scope="session")
def active_calibration_surface_data():
    shared_pars = {
        "bead_diameter": 1.03,
        "viscosity": 1.1e-3,
        "temperature": 25,
        "rho_sample": 997.0,
        "rho_bead": 1040.0,
        "distance_to_surface": 1.03 / 2 + 500e-3,
    }

    sim_pars = {
        "sample_rate": 78125,
        "stiffness": 0.1,
        "pos_response_um_volt": 0.618,
        "driving_sinusoid": (500, 31.95633),
        "diode": (0.4, 15000),
    }

    np.random.seed(10071985)
    volts, stage = generate_active_calibration_test_data(
        10, hydrodynamically_correct=True, **sim_pars, **shared_pars
    )

    active_pars = {
        "force_voltage_data": volts,
        "driving_data": stage,
        "sample_rate": 78125,
        "driving_frequency_guess": 32,
    }

    return shared_pars, sim_pars, active_pars


@pytest.fixture()
def calibration_data():
    np.random.seed(1337)
    num_samples = 100000
    dummy_voltage = np.random.normal(size=num_samples)
    dummy_nano = np.sin(2.0 * np.pi * 17 * np.arange(num_samples) / 78125)
    return dummy_voltage, dummy_nano


@pytest.fixture(scope="module")
def reference_calibration_result():
    data = np.load(os.path.join(os.path.dirname(__file__), "data/reference_spectrum.npz"))
    reference_spectrum = data["arr_0"]
    model = PassiveCalibrationModel(4.4, temperature=20, viscosity=0.001002)
    reference_spectrum = psc.calculate_power_spectrum(
        reference_spectrum, sample_rate=78125, num_points_per_block=100, fit_range=(100.0, 23000.0)
    )
    ps_calibration = psc.fit_power_spectrum(
        power_spectrum=reference_spectrum, model=model, bias_correction=False
    )

    return ps_calibration, model, reference_spectrum
