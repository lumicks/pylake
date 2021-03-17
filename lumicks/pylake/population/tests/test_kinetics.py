import pytest
import numpy as np
from matplotlib.testing.decorators import cleanup
import matplotlib.pyplot as plt
from lumicks.pylake.population.kinetics import ExponentialDistribution
from lumicks.pylake.population.detail.random import temp_seed


def draw_exponential_sample(tau, size, t_min, seed):
    N = 0
    data = []
    with temp_seed(seed):
        while N < size:
            tmp = np.random.exponential(tau, size=size)
            tmp = tmp[np.argwhere(tmp > t_min)]
            N += tmp.size
            data.append(tmp.squeeze())
    data = np.hstack(data)
    return data[:size]


@pytest.fixture(params=[(0.2, 0.0), (0.2, 0.1)])
def exp_sample(request):
    tau, t_min = request.param
    data = draw_exponential_sample(tau, 1000, t_min, 123)
    return data, 1/tau, t_min


@pytest.fixture
def exp_small():
    tau = 0.2
    t_min = 0.0
    data = draw_exponential_sample(tau, 50, t_min, 123)
    return data, 1/tau, t_min


def test_exponential_fit(exp_sample):
    data, k, t_min = exp_sample

    # without correction for minimum data acquisition time
    e = ExponentialDistribution(data, t_min=0)
    result = np.allclose(k, e.rate, atol=0.2)
    assert result if t_min == 0 else (not result)

    # with correction
    e = ExponentialDistribution(data, t_min=t_min)
    assert np.allclose(k, e.rate, atol=0.2)

    ci = e.ci(0.95)
    assert ci[0] < e.rate < ci[1]
    with pytest.raises(ValueError):
        e.ci(5)


def test_exp_pdf(exp_small):
    data, k, t_min = exp_small
    with temp_seed(234):
        e = ExponentialDistribution(data, t_min=t_min)
    assert np.allclose(e.pdf(1/k), 1.829696)


@cleanup
def test_exponential_plots(exp_small):
    data, k, t_min = exp_small
    e = ExponentialDistribution(data, t_min=t_min)

    e.hist()
    plt.close()
