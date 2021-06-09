import pytest
from lumicks.pylake.fitting.profile_likelihood import clamp_step, StepConfig, do_step
import numpy as np


@pytest.mark.parametrize("origin,step,lb,ub,result", [
    # Positive quadrant
    (np.array([1, 1]), np.array([-2, -4]), np.array([0, 0]), np.array([2, 2]), np.array([0.5, 0.0])),
    (np.array([1, 1]), np.array([-4, -4]), np.array([0, 0]), np.array([2, 2]), np.array([0, 0])),
    (np.array([1, 1]), np.array([4, 4]), np.array([0, 0]), np.array([2, 2]), np.array([2, 2])),
    (np.array([1, 1]), np.array([2, 4]), np.array([0, 0]), np.array([2, 2]), np.array([1.5, 2])),

    (np.array([1, 1]), np.array([-2, 4]), np.array([0, 0]), np.array([2, 2]), np.array([0.5, 2])),
    (np.array([1, 1]), np.array([2, -4]), np.array([0, 0]), np.array([2, 2]), np.array([1.5, 0])),

    (np.array([1, 1]), np.array([0, .5]), np.array([0, 0]), np.array([2, 2]), np.array([1.0, 1.5])),
    (np.array([1, 1]), np.array([.5, 0]), np.array([0, 0]), np.array([2, 2]), np.array([1.5, 1])),
    (np.array([1, 1]), np.array([-.5, 0]), np.array([0, 0]), np.array([2, 2]), np.array([0.5, 1])),
    (np.array([1, 1]), np.array([-.5, -.5]), np.array([0, 0]), np.array([2, 2]), np.array([0.5, 0.5])),
    (np.array([3, 3]), np.array([-.5, 0.0]), np.array([2, 2]), np.array([4, 4]), np.array([2.5, 3.0])),
    (np.array([3, 3]), np.array([-10.0, 0.0]), np.array([2, 2]), np.array([4, 4]), np.array([2.0, 3.0])),

    # Negative quadrant
    (np.array([-1, -1]), np.array([-2, -4]), np.array([-2, -2]), np.array([0, 0]), np.array([-1.5, -2])),
    (np.array([-1, -1]), np.array([-4, -4]), np.array([-2, -2]), np.array([0, 0]), np.array([-2, -2])),
    (np.array([-1, -1]), np.array([4, 4]), np.array([-2, -2]), np.array([0, 0]), np.array([0, 0])),
    (np.array([-1, -1]), np.array([2, 4]), np.array([-2, -2]), np.array([0, 0]), np.array([-0.5, 0])),

    (np.array([-1, -1]), np.array([-2, 4]), np.array([-2, -2]), np.array([0, 0]), np.array([-1.5, 0.0])),
    (np.array([-1, -1]), np.array([2, -4]), np.array([-2, -2]), np.array([0, 0]), np.array([-0.5, -2])),

    # Both quadrants
    (np.array([-1, -1]), np.array([-2, -4]), np.array([-2, -2]), np.array([2, 2]), np.array([-1.5, -2])),
    (np.array([-1, -1]), np.array([-4, -4]), np.array([-2, -2]), np.array([2, 2]), np.array([-2, -2])),
    (np.array([-1, -1]), np.array([4, 4]), np.array([-2, -2]), np.array([2, 2]), np.array([2, 2])),
    (np.array([-1, -1]), np.array([2, 4]), np.array([-2, -2]), np.array([2, 2]),
     np.array([-1.0 + 2.0 * (3.0 / 4.0), 2])),

    (np.array([-1, -1]), np.array([-2, 4]), np.array([-2, -2]), np.array([2, 2]), np.array([-2.0, 1.0])),
    (np.array([-1, -1]), np.array([2, -4]), np.array([-2, -2]), np.array([2, 2]), np.array([-0.5, -2.0])),
])
def test_clamp_vector(origin, step, lb, ub, result):
    np.testing.assert_allclose(clamp_step(origin, step, lb, ub)[0], result)


cfg = StepConfig(min_abs_step=1e-4,
                 max_abs_step=2,
                 step_factor=2,
                 min_chi2_step_size=1,
                 max_chi2_step_size=10,
                 lower_bounds=np.array([-2, -2]),
                 upper_bounds=np.array([4, 4]))


class FlippingCost:
    def __init__(self):
        self.flip = 0.0

    def __call__(self, x):
        self.flip = 25.0 - self.flip
        return self.flip


@pytest.mark.parametrize("chi2_function,start_pos,step_sign,target_step_size,target_p_trial", [
    (lambda x: 5, np.array([2, 2]), 1, cfg.max_abs_step, np.array([2.0 + cfg.max_abs_step, 2.0 + cfg.max_abs_step])),
    (lambda x: 8, np.array([2, 2]), 1, 1, np.array([3.0, 3.0])),
    (lambda x: 8, np.array([4, 4]), 1, 1, np.array([4.0, 4.0])),
    (lambda x: 8, np.array([2, 2]), -1, 1, np.array([1.0, 1.0])),
    (lambda x: 8, np.array([-1.0, -2.0]), -1, 1, np.array([-1.0, -2.0])),
    # When the cost alternates, it should only decrease the step size once
    (FlippingCost(), np.array([2, 2]), 1, .5, np.array([2.5, 2.5])),
])
def test_stepper(chi2_function, start_pos, step_sign, target_step_size, target_p_trial):
    step_size, p_trial = do_step(chi2_function=chi2_function,
                                 step_direction_function=lambda: np.array([1, 1]),
                                 chi2_last=5,
                                 parameter_vector=start_pos,
                                 current_step_size=1,
                                 step_sign=step_sign,
                                 step_config=cfg)

    assert step_size == target_step_size
    np.testing.assert_allclose(p_trial, target_p_trial)


def test_minimum_step():
    # Here we provide the step size determination algorithm with a fixed increase of the cost by
    # chi2_function - chi2_last = 20 no matter what parameter it tries. Given that our maximum
    # step size to take is 10 (see max_chi2_step_size in the cfg), this will result in the step size
    # shrinking to the minimum step size. In a non-pathological case, a smaller step size would
    # result in a smaller chi2 increase.
    with pytest.warns(RuntimeWarning, match="Warning: Step size set to minimum step size."):
        step_size, p_trial = do_step(chi2_function=lambda x: 25,
                                     step_direction_function=lambda: np.array([1, 1]),
                                     chi2_last=5,
                                     parameter_vector=np.array([2, 2]),
                                     current_step_size=1,
                                     step_sign=1,
                                     step_config=cfg)
        assert step_size == cfg.min_abs_step
        np.testing.assert_allclose(p_trial, np.array([2.0 + cfg.min_abs_step, 2.0 + cfg.min_abs_step]))
