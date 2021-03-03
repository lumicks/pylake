import numpy as np
from pathlib import Path


def generate_parameters(n_states):
    return {
        "initial_state_prob": np.ones(n_states) / n_states,
        "transition_prob": np.eye(n_states) * 9 + np.ones((n_states, n_states)),
        "means": np.arange(n_states) + 10,
        "st_devs": np.full(n_states, 0.1),
    }


def generate_trace(n_states, initial_state_prob, transition_prob, means, st_devs, n_frames=100, seed=123):
    """Generate a time trace from HMM parameters.

    Parameters
    ----------
    n_states : int
        number of states
    initial_state_prob : initial state probability vector
        array of shape (n_states, )
    transition_prob : state transition probability matrix
        array of shape (n_states, n_states)
    means : state means
        array of shape (n_states, )
    st_devs : state standard deviations
        array of shape (n_states, )
    n_frames : int
        trace length
    seed : int
        random seed to use for generator
    """
    np.random.seed(seed)

    # make sure vectors are normalized
    initial_state_prob = initial_state_prob / initial_state_prob.sum()
    transition_prob = transition_prob / transition_prob.sum(axis=1).reshape((-1, 1))

    # draw state path
    sp = np.zeros(n_frames, dtype=int)
    sp[0] = np.random.choice(n_states, 1, replace=False, p=initial_state_prob).squeeze()
    for t in range(1, n_frames):
        sp[t] = np.random.choice(n_states, 1, replace=False, p=transition_prob[sp[t - 1]]).squeeze()

    # draw observation vector
    y = np.random.normal(loc=means[sp], scale=st_devs[sp])

    # reset RNG
    np.random.seed(None)
    return y, sp


if __name__ == "__main__":
    save_path = Path(__file__).parent / "trace_data.npz"

    data = {}
    for n_states in (2, 3, 4):
        param = generate_parameters(n_states)
        y, sp = generate_trace(n_states, n_frames=100, seed=123, **param)

        data[f"y_{n_states}"] = y
        data[f"sp_{n_states}"] = sp
        for key, val in param.items():
            data[f"{key}_{n_states}"] = param[key]

    np.savez(save_path, **data)
