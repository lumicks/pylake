import numpy as np

def generate_fdcurve_with_baseline_offset():
    # generate high frequency data
    distance_hf = np.linspace(1, 10, 500)                       # distance, HF
    time_hf = np.arange(0, 3*distance_hf.size, 3) + 100         # timestamps, HF
    true_f_hf = np.exp(0.5 * distance_hf)                       # true force, HF

    # make baseline
    p = [-1e-5, 1e-4, 1e-3, 0.005, 0.3, -5, 30] # baseline polynomial coefficients
    bl = np.polyval(p, distance_hf)                             # baseline
    obs_f_hf = true_f_hf + bl                                   # observed force, HF

    # manually downsample to low frequency like BL
    win = 5
    rng = range(0, distance_hf.size-win, win)
    downsample = lambda r, w, hf: np.array([np.mean(hf[start:start+w]) for start in r])

    time_lf = np.array([time_hf[start+win] for start in rng])   # timestamps, LF
    distance_lf = downsample(rng, win, distance_hf)             # distance, LF
    true_f_lf = downsample(rng, win, true_f_hf)                 # true force, LF
    obs_f_lf = downsample(rng, win, obs_f_hf)                   # observed force, LF

    data = {
        "HF": {
            "time": time_hf,
            "true_force": true_f_hf,
            "obs_force": obs_f_hf,
        },
        "LF": {
            "time": time_lf,
            "true_force": true_f_lf,
            "obs_force": obs_f_lf,
            "distance": distance_lf
        }
    }

    return p, data
