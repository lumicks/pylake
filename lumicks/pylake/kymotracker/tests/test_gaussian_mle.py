from numpy.core.shape_base import stack
import pytest
import numpy as np
from pathlib import Path

from lumicks.pylake.kymotracker.detail.gaussian_mle import _extract_params, poisson_log_likelihood, poisson_log_likelihood_jacobian
from lumicks.pylake.kymotracker.detail.gaussian_mle import run_gaussian_mle
from lumicks.pylake.fitting.detail.derivative_manipulation import numerical_jacobian
from .data.generate_gaussian_track_data import generate_peak


class GaussianImage:

    def __init__(self, name):
        self.name = name
        data = np.load(Path(__file__).parent / "data/gaussian_track_data.npz")
        self.position = data["position"]
        self.pixel_size = data["pixel_size"]
        self.n_frames = data["n_frames"]

        self.amplitudes = data[f"{name}_amplitude"]
        self.centers = data[f"{name}_center"]
        self.scales = data[f"{name}_scale"]
        self.offsets = data[f"{name}_offset"]

        self.expectation = data[f"{name}_expectation"]
        self.photon_count = data[f"{name}_photon_count"]

    def get_frame(self, index):
        return self.photon_count[:, index][:, np.newaxis]

    def get_frame_params(self, index):
        return np.hstack(
            [self.amplitudes[index], self.centers[index], self.scales[index], self.offsets[index]]
        )

    def stack_params(self, shared_var=False, shared_off=False):
        return np.hstack(
            [
                self.amplitudes,
                self.centers,
                self.scales[0] if shared_var else self.scales,
                self.offsets[0] if shared_off else self.offsets
            ]
        )


@pytest.fixture(scope="module")
def high_intensity():
    return GaussianImage("high_intensity")


@pytest.fixture(scope="module")
def variable_params():
    return GaussianImage("variable_params")


def test_parameter_extraction():
    def check_parameters(p):
        amplitude, center, scale, offset = p
        strip = lambda x: set([s for s in list("".join(x)) if not s.isdigit()])
        assert strip(amplitude) == {"a"}
        assert strip(center) == {"m"}
        assert strip(scale) == {"s"}
        assert strip(offset) == {"b"}

    # no shared parameters
    check_parameters(_extract_params(np.array(["a1", "m1", "s1", "b1"]),
                                     n_frames=1, shared_variance=False, shared_offset=False))
    check_parameters(_extract_params(np.array(["a1", "a2", "m1", "m2", "s1", "s2", "b1", "b2"]),
                                     n_frames=2, shared_variance=False, shared_offset=False))
    check_parameters(_extract_params(np.array(["a1", "a2", "a3", "m1", "m2", "m3", "s1", "s2", "s3", "b1", "b2", "b3"]),
                                     n_frames=3, shared_variance=False, shared_offset=False))

    # shared variance, local offset
    check_parameters(_extract_params(np.array(["a1", "m1", "s", "b1"]),
                                     n_frames=1, shared_variance=True, shared_offset=False))
    check_parameters(_extract_params(np.array(["a1", "a2", "m1", "m2", "s", "b1", "b2"]),
                                     n_frames=2, shared_variance=True, shared_offset=False))
    check_parameters(_extract_params(np.array(["a1", "a2", "a3", "m1", "m2", "m3", "s", "b1", "b2", "b3"]),
                                     n_frames=3, shared_variance=True, shared_offset=False))

    # local variance, shared offset
    check_parameters(_extract_params(np.array(["a1", "m1", "s1", "b"]),
                                     n_frames=1, shared_variance=False, shared_offset=True))
    check_parameters(_extract_params(np.array(["a1", "a2", "m1", "m2", "s1", "s2", "b"]),
                                     n_frames=2, shared_variance=False, shared_offset=True))
    check_parameters(_extract_params(np.array(["a1", "a2", "a3", "m1", "m2", "m3", "s1", "s2", "s3", "b"]),
                                     n_frames=3, shared_variance=False, shared_offset=True))

    # shared variance, shared offset
    check_parameters(_extract_params(np.array(["a1", "m1", "s", "b"]),
                                     n_frames=1, shared_variance=True, shared_offset=True))
    check_parameters(_extract_params(np.array(["a1", "a2", "m1", "m2", "s", "b"]),
                                     n_frames=2, shared_variance=True, shared_offset=True))
    check_parameters(_extract_params(np.array(["a1", "a2", "a3", "m1", "m2", "m3", "s", "b"]),
                                     n_frames=3, shared_variance=True, shared_offset=True))


def test_likelihood(high_intensity):
    image = high_intensity
    multipliers = [1, 0.9, 1.1]

    # single line
    results = (-75718.35378780862,
               -70920.84658584111,
               -72526.71503572378)
    frame_params = image.get_frame_params(0)
    line = image.get_frame(0)
    for m, result in zip(multipliers, results):
        fitted_result = poisson_log_likelihood(frame_params * m, image.position, line)
        assert np.allclose(result, fitted_result)

    # full track - all permutations of shared variance/offset have same likelihood for given parameters
    results = (-230682.14104404833,
               -216398.08519266717,
               -220792.27484096336)

    def check_results(shared_var, shared_off):
        image_params = image.stack_params(shared_var=shared_var, shared_off=shared_off)
        for m, result in zip(multipliers, results):
            fitted_result = poisson_log_likelihood(image_params * m, image.position, image.photon_count, shared_var, shared_off)
            assert np.allclose(result, fitted_result)

    check_results(False, False) # local variance, local offset
    check_results(True, False)  # global variance, local offset
    check_results(False, True)  # local variance, global offset
    check_results(True, True)   # global variance, global offset


def test_likelihood_derivative(high_intensity):
    image = high_intensity
    multipliers = [1, 0.99, 1.01]

    # single line
    results = ([1.00286872e-01, -5.77974134e+02,  4.00093218e+02,  4.18406691e+00],
               [-7.28745643e-02, -3.35159172e+03, -1.33296883e+03,  2.08948522e+00],
               [2.72565434e-01, 2.09378745e+03, 1.71048391e+03, 5.63057317e+00])
    frame_params = image.get_frame_params(0)
    line = image.get_frame(0)
    for m, result in zip(multipliers, results):
        fitted_result = poisson_log_likelihood_jacobian(frame_params * m, image.position, line)
        assert np.allclose(result, fitted_result)

    # full track
    def check_results(shared_var, shared_off, results):
        image_params = image.stack_params(shared_var=shared_var, shared_off=shared_off)
        for m, result in zip(multipliers, results):
            fitted_result = poisson_log_likelihood_jacobian(image_params * m, image.position, image.photon_count, shared_var, shared_off)
            assert np.allclose(result, fitted_result)

    # local variance, local offset
    results = ([ 1.00286872e-01, -2.71443302e-02, -1.06620967e-01,
                -5.77974134e+02, -1.18050418e+02,  2.19611876e+02,
                4.00093218e+02,  4.11564317e+02, -4.49203725e+02,
                4.18406691e+00,  1.25746852e+00,  8.64946438e-01],
               [-7.28745643e-02, -2.01842173e-01, -2.82170847e-01,
                -3.35159172e+03, -2.92746767e+03, -2.61099779e+03,
                -1.33296883e+03, -1.26158509e+03, -2.10382382e+03,
                2.08948522e+00, -6.79757848e-01, -1.03931316e+00],
               [2.72565434e-01, 1.46542734e-01, 6.80101201e-02,
                2.09378745e+03, 2.58669386e+03, 2.94555809e+03,
                1.71048391e+03, 1.65505607e+03, 7.77737822e+02,
                5.63057317e+00, 2.62284587e+00, 2.11636547e+00])
    check_results(False, False, results)

    # full track - shared variance, local offset
    results = ([1.00286872e-01, -2.71443302e-02, -1.06620967e-01,
                -5.77974134e+02, -1.18050418e+02,  2.19611876e+02,
                3.62453810e+02,
                4.18406691e+00, 1.25746852e+00,  8.64946438e-01],
               [-7.28745643e-02, -2.01842173e-01, -2.82170847e-01,
                -3.35159172e+03, -2.92746767e+03, -2.61099779e+03,
                -4.69837773e+03,
                2.08948522e+00, -6.79757848e-01, -1.03931316e+00],
               [2.72565434e-01, 1.46542734e-01, 6.80101201e-02,
                2.09378745e+03, 2.58669386e+03, 2.94555809e+03,
                4.14327780e+03,
                5.63057317e+00, 2.62284587e+00, 2.11636547e+00])
    check_results(True, False, results)

    # full track - local variance, shared offset
    results = ([1.00286872e-01, -2.71443302e-02, -1.06620967e-01,
                -5.77974134e+02, -1.18050418e+02,  2.19611876e+02,
                4.00093218e+02,  4.11564317e+02, -4.49203725e+02,
                6.30648187e+00],
               [-7.28745643e-02, -2.01842173e-01, -2.82170847e-01,
                -3.35159172e+03, -2.92746767e+03, -2.61099779e+03,
                -1.33296883e+03, -1.26158509e+03, -2.10382382e+03,
                3.70414207e-01],
               [2.72565434e-01, 1.46542734e-01, 6.80101201e-02,
                2.09378745e+03, 2.58669386e+03, 2.94555809e+03,
                1.71048391e+03, 1.65505607e+03, 7.77737822e+02,
                1.03697845e+01])
    check_results(False, True, results)

    # full track - shared variance, shared offset
    results = ([1.00286872e-01, -2.71443302e-02, -1.06620967e-01,
                -5.77974134e+02, -1.18050418e+02,  2.19611876e+02,
                3.62453810e+02,  6.30648187e+00],
               [-7.28745643e-02, -2.01842173e-01, -2.82170847e-01,
                -3.35159172e+03, -2.92746767e+03, -2.61099779e+03,
                -4.69837773e+03,  3.70414207e-01],
               [2.72565434e-01, 1.46542734e-01, 6.80101201e-02,
                2.09378745e+03, 2.58669386e+03, 2.94555809e+03,
                4.14327780e+03, 1.03697845e+01])
    check_results(True, True, results)


def test_jacobian(variable_params):
    image = variable_params

    # single-line fit
    f = lambda p: np.array([poisson_log_likelihood(p, image.position, image.get_frame(0), False, False)])
    try_params = image.get_frame_params(0) * [0.8, 1.1, 1.25, 0.85]

    num_result = numerical_jacobian(f, try_params, dx=1e-6).squeeze()
    test_result = poisson_log_likelihood_jacobian(try_params, image.position, image.get_frame(0), False, False).squeeze()
    assert np.allclose(num_result, test_result)

    # simultaneous image fit, all local parameters
    f = lambda p: np.array([poisson_log_likelihood(p, image.position, image.photon_count, False, False)])
    try_params = np.hstack([image.amplitudes * [0.8, 1.1, 0.7],
                            image.centers * [0.9, 0.7, 1.1],
                            image.scales * [0.7, 1.2, 0.8],
                            image.offsets * [0.7, 0.8, 1.3]])
    num_result = numerical_jacobian(f, try_params, dx=1e-6).squeeze()
    test_result = poisson_log_likelihood_jacobian(try_params, image.position, image.photon_count, False, False).squeeze()
    assert np.allclose(num_result, test_result)

    # shared variance
    f = lambda p: np.array([poisson_log_likelihood(p, image.position, image.photon_count, True, False)])
    try_params = np.hstack([image.amplitudes * [0.8, 1.1, 0.7],
                            image.centers * [0.9, 0.7, 1.1],
                            image.scales[0] * 1.2,
                            image.offsets * [0.7, 0.8, 1.3]])
    num_result = numerical_jacobian(f, try_params, dx=1e-6).squeeze()
    test_result = poisson_log_likelihood_jacobian(try_params, image.position, image.photon_count, True, False).squeeze()
    assert np.allclose(num_result, test_result)

    # shared offset
    f = lambda p: np.array([poisson_log_likelihood(p, image.position, image.photon_count, False, True)])
    try_params = np.hstack([image.amplitudes * [0.8, 1.1, 0.7],
                            image.centers * [0.9, 0.7, 1.1],
                            image.scales * [0.7, 1.2, 0.8],
                            image.offsets[2] * 0.7])
    num_result = numerical_jacobian(f, try_params, dx=1e-6).squeeze()
    test_result = poisson_log_likelihood_jacobian(try_params, image.position, image.photon_count, False, True).squeeze()
    assert np.allclose(num_result, test_result)

    # simultaneous image fit, all local parameters
    f = lambda p: np.array([poisson_log_likelihood(p, image.position, image.photon_count, True, True)])
    try_params = np.hstack([image.amplitudes * [0.8, 1.1, 0.7],
                            image.centers * [0.9, 0.7, 1.1],
                            image.scales[2] * 0.8,
                            image.offsets[0] * 1.3])
    num_result = numerical_jacobian(f, try_params, dx=1e-6).squeeze()
    test_result = poisson_log_likelihood_jacobian(try_params, image.position, image.photon_count, True, True).squeeze()
    assert np.allclose(num_result, test_result)


def test_mle_fit(high_intensity):
    image = high_intensity

    # single line
    result = run_gaussian_mle(image.position, image.get_frame(0), image.pixel_size)
    assert np.allclose(result, [1.49401257e+03, 2.60554393e+00, 3.49511656e-01, 1.67587646e+00])

    result = run_gaussian_mle(image.position, image.photon_count, image.pixel_size, shared_variance=False, shared_offset=False)
    assert np.allclose(result, [1.49401689e+03, 1.54401473e+03, 1.51001867e+03,
                                2.60554250e+00, 2.60111298e+00, 2.59794974e+00,
                                3.49237600e-01, 3.45392909e-01, 3.50812234e-01,
                                1.90962878e+00, 1.80283161e+00, 1.82445477e+00])

    result = run_gaussian_mle(image.position, image.photon_count, image.pixel_size, shared_variance=True, shared_offset=False)
    assert np.allclose(result, [1.49401683e+03, 1.54401469e+03, 1.51001858e+03,
                                2.60556814e+00, 2.60110016e+00, 2.59797177e+00,
                                3.48463940e-01,
                                1.94250547e+00, 1.79896740e+00, 1.82080899e+00,])

    result = run_gaussian_mle(image.position, image.photon_count, image.pixel_size, shared_variance=False, shared_offset=True)
    assert np.allclose(result, [1.49400451e+03, 1.54400393e+03, 1.51000499e+03,
                                2.60554395e+00, 2.60111698e+00, 2.59795053e+00,
                                3.49304763e-01, 3.45339881e-01, 3.50787430e-01,
                                1.85155803e+00])

    result = run_gaussian_mle(image.position, image.photon_count, image.pixel_size, shared_variance=True, shared_offset=True)
    assert np.allclose(result, [1.49400210e+03, 1.54400182e+03, 1.51000233e+03,
                                2.60554529e+00, 2.60111082e+00, 2.59794286e+00,
                                3.48476245e-01, 1.84943345e+00])
