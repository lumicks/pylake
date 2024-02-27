import numpy as np
from cachetools import cached

from lumicks.pylake.kymo import _kymo_from_array
from lumicks.pylake.kymotracker.kymotracker import KymoTrack, KymoTrackGroup


def _simulate_diffusion_1d(diffusion_constant, steps, dt, observation_noise):
    """Simulate from a Wiener process

    Parameters
    ----------
    diffusion_constant : float
        Diffusion constant.
    steps : int
        Number of steps to simulate.
    dt : float
        Time step.
    observation_noise : float
        Standard deviation of the observation noise.
    """

    def simulate_wiener(sigma, num_steps, time_step):
        return np.cumsum(np.random.normal(0, sigma * np.sqrt(time_step), size=(num_steps,)))

    return simulate_wiener(np.sqrt(2.0 * diffusion_constant), steps, dt) + np.random.normal(
        0, observation_noise, (steps,)
    )


# We use a cache to ensure that KymoTrackGroup with the same properties end up referring to the same
# kymograph. This allows them to be added.
@cached(cache={})
def _get_blank_kymo(*, line_time_seconds):
    """Generate kymo with particular properties"""

    kymo = _kymo_from_array(
        np.zeros((0, 0)), "r", line_time_seconds=line_time_seconds, pixel_size_um=1.0
    )
    kymo._motion_blur_constant = 0
    return kymo


def simulate_diffusive_tracks(
    diffusion_constant,
    steps,
    dt,
    *,
    observation_noise=0,
    num_tracks=1,
):
    """Generate a KymoTrackGroup of pure diffusive traces.

    Parameters
    ----------
    diffusion_constant : float
        Diffusion constant.
    steps : int
        Number of steps to simulate.
    dt : float
        Time step.
    observation_noise : float
        Standard deviation of the observation noise.
    num_tracks : int
        Number of tracks to simulate.
    """

    # We need a Kymo so that half the API doesn't break.
    blank_kymo = _get_blank_kymo(line_time_seconds=dt)
    time_idx = np.arange(0, steps)

    return KymoTrackGroup(
        [
            KymoTrack(
                time_idx=time_idx,
                localization=_simulate_diffusion_1d(
                    diffusion_constant, steps, dt, observation_noise
                ),
                kymo=blank_kymo,
                channel="red",
                minimum_observable_duration=blank_kymo.line_time_seconds,
            )
            for _ in range(num_tracks)
        ]
    )
