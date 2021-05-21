import numpy as np

from lumicks.pylake.kymotracker.detail.gaussian_mle import _extract_params, poisson_log_likelihood, poisson_log_likelihood_jacobian
from lumicks.pylake.kymotracker.detail.gaussian_mle import run_gaussian_mle
from lumicks.pylake.fitting.detail.derivative_manipulation import numerical_jacobian


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
    position, pixel_size, line, photon_count, true_params, image_params = high_intensity
    n_frames = photon_count.shape[1]
    multipliers = [1, 0.9, 1.1]

    # single line
    results = (-75718.35378780862,
               -70920.84658584111,
               -72526.71503572378)
    for m, result in zip(multipliers, results):
        fitted_result = poisson_log_likelihood(true_params*m, position, line)
        assert np.allclose(result, fitted_result)

    # full track - all permutations of shared variance/offset have same likelihood for given parameters
    results = (-230682.14104404833,
               -216398.08519266717,
               -220792.27484096336)

    def check_results(shared_var, shared_off):
        idx = (shared_var, shared_off)
        for m, result in zip(multipliers, results):
            fitted_result = poisson_log_likelihood(image_params[idx] * m, position, photon_count, *idx)
            assert np.allclose(result, fitted_result)

    check_results(False, False) # local variance, local offset
    check_results(True, False)  # global variance, local offset
    check_results(False, True)  # local variance, global offset
    check_results(True, True)   # global variance, global offset


def test_likelihood_derivative(high_intensity):
    position, pixel_size, line, photon_count, true_params, image_params = high_intensity
    n_frames = photon_count.shape[1]
    multipliers = [1, 0.99, 1.01]

    # single line
    results = ([1.00286872e-01, -5.77974134e+02,  4.00093218e+02,  4.18406691e+00],
               [-7.28745643e-02, -3.35159172e+03, -1.33296883e+03,  2.08948522e+00],
               [2.72565434e-01, 2.09378745e+03, 1.71048391e+03, 5.63057317e+00])
    for m, result in zip(multipliers, results):
        fitted_result = poisson_log_likelihood_jacobian(true_params*m, position, line)
        assert np.allclose(result, fitted_result)

    # full track
    def check_results(shared_var, shared_off, result):
        idx = (shared_var, shared_off)
        for m, result in zip(multipliers, results):
            fitted_result = poisson_log_likelihood_jacobian(image_params[idx] * m, position, photon_count, *idx)
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


def test_jacobian(high_intensity):
    position, pixel_size, line, photon_count, true_params, image_params = high_intensity
    n_frames = photon_count.shape[1]
    multiplier = 0.999

    # single line
    f = lambda p: np.array([poisson_log_likelihood(p, position, line, False, False)])
    num_result = numerical_jacobian(f, true_params * multiplier, dx=1e-6).squeeze()
    test_result = poisson_log_likelihood_jacobian(true_params * multiplier, position, line, False, False).squeeze()
    assert np.allclose(num_result, test_result, rtol=1e-4)

    # full track
    def check_result(shared_var, shared_off):
        f = lambda p: np.array([poisson_log_likelihood(p, position, photon_count, shared_var, shared_off)])
        idx = (shared_var, shared_off)
        num_result = numerical_jacobian(f, image_params[idx] * multiplier, dx=1e-6).squeeze()
        test_result = poisson_log_likelihood_jacobian(image_params[idx] * multiplier, position, photon_count, shared_var, shared_off).squeeze()
        assert np.allclose(num_result, test_result, rtol=2e-4)
        print(idx)

    check_result(False, False)  # local variance, local offset
    check_result(True, False)   # global variance, local offset
    check_result(False, True)   # local variance, global offset
    check_result(True, True)    # global variance, global offset


def test_mle_fit(high_intensity):
    position, pixel_size, line, photon_count, true_params, image_params = high_intensity

    # single line
    result = run_gaussian_mle(position, line, pixel_size)
    assert np.allclose(result, [1.49401257e+03, 2.60554393e+00, 3.49511656e-01, 1.67587646e+00])

    result = run_gaussian_mle(position, photon_count, pixel_size, shared_variance=False, shared_offset=False)
    assert np.allclose(result, [1.49401689e+03, 1.54401473e+03, 1.51001867e+03,
                                2.60554250e+00, 2.60111298e+00, 2.59794974e+00,
                                3.49237600e-01, 3.45392909e-01, 3.50812234e-01,
                                1.90962878e+00, 1.80283161e+00, 1.82445477e+00])

    result = run_gaussian_mle(position, photon_count, pixel_size, shared_variance=True, shared_offset=False)
    assert np.allclose(result, [1.49401689e+03, 1.54401474e+03, 1.51001865e+03,
                                2.60555100e+00, 2.60111617e+00, 2.59795441e+00,
                                3.48479939e-01,
                                1.91080564e+00, 1.80183763e+00, 1.82375124e+00])

    result = run_gaussian_mle(position, photon_count, pixel_size, shared_variance=False, shared_offset=True)
    assert np.allclose(result, [1.49400630e+03, 1.54400549e+03, 1.51000696e+03,
                                2.60553447e+00, 2.60111017e+00, 2.59794655e+00,
                                3.49317957e-01, 3.45348040e-01, 3.50791137e-01,
                                1.84919457e+00])

    result = run_gaussian_mle(position, photon_count, pixel_size, shared_variance=True, shared_offset=True)
    assert np.allclose(result, [1.49400304e+03, 1.54400264e+03, 1.51000337e+03,
                                2.60554802e+00, 2.60111993e+00, 2.59795325e+00,
                                3.48478417e-01, 1.84889360e+00])
