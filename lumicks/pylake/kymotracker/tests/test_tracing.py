from lumicks.pylake.kymotracker.detail.scoring_functions import build_score_matrix
from lumicks.pylake.kymotracker.detail.trace_line_2d import KymoLine, assign_to_lines
import pytest
import numpy as np


def test_score_matrix():
    lines = [KymoLine([0], [3])]
    unique_coordinates = np.arange(0, 7)
    unique_times = np.arange(1, 7)

    positions = np.array([])
    times = np.array([])
    for t in unique_times:
        positions = np.hstack((positions, unique_coordinates))
        times = np.hstack((times, t * np.ones(len(unique_coordinates))))

    # No velocity
    matrix = np.reshape(build_score_matrix(lines, times, positions, vel=0, sigma=.5, diffusion=.5, sigma_cutoff=2),
                        ((len(unique_times), -1)))
    reference = [
        [-np.inf, -np.inf, -1.0, -0.0, -1.0, -np.inf, -np.inf],
        [-np.inf, -2.745166004060959, -0.6862915010152397, -0.0, -0.6862915010152397, -2.745166004060959, -np.inf],
        [-np.inf, -2.1435935394489816, -0.5358983848622454, -0.0, -0.5358983848622454, -2.1435935394489816, -np.inf],
        [-np.inf, -1.7777777777777777, -0.4444444444444444, -0.0, -0.4444444444444444, -1.7777777777777777, -np.inf],
        [
            -3.4376941012509463,
            -1.5278640450004204,
            -0.3819660112501051,
            -0.0,
            -0.3819660112501051,
            -1.5278640450004204,
            -3.4376941012509463,
        ],
        [
            -3.0254695407844476,
            -1.3446531292375323,
            -0.3361632823093831,
            -0.0,
            -0.3361632823093831,
            -1.3446531292375323,
            -3.0254695407844476,
        ],
    ]
    assert np.allclose(matrix, reference)

    # With velocity
    matrix = np.reshape(build_score_matrix(lines, times, positions, vel=1, sigma=.5, diffusion=.5, sigma_cutoff=2),
                        (len(unique_times), -1))
    reference = [
        [-np.inf, -np.inf, -np.inf, -1.0, -0.0, -1.0, -np.inf],
        [-np.inf, -np.inf, -np.inf, -2.745166004060959, -0.6862915010152397, -0.0, -0.6862915010152397],
        [-np.inf, -np.inf, -np.inf, -np.inf, -2.1435935394489816, -0.5358983848622454, -0.0],
        [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -1.7777777777777777, -0.4444444444444444],
        [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -3.4376941012509463, -1.5278640450004204],
        [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -3.0254695407844476],
    ]
    assert np.allclose(matrix, reference)


def test_score_based_tracing():
    # Test addition order
    lines = [KymoLine([0], [0])]
    score = np.array([[-2.0, -1.0, 0.0]])
    time = [0.0, 1.0, 2.0]
    coordinate = [0.0, 1.0, 2.0]
    lines, time = assign_to_lines(score, lines, time, coordinate, seen=10)
    assert np.allclose(lines[0].coordinate, [0, 2])
    assert np.allclose(time, [0, 1, 10])

    # Test addition order (limited by line count)
    lines = [KymoLine([0], [0]), KymoLine([1], [1])]
    score = np.array([[2.0, 1.0, 0.0], [4.0, 0.5, 0.0]])
    time = [0.0, 1.0, 2.0]
    coordinate = [0.0, 1.0, 2.0]
    lines, time = assign_to_lines(score, lines, time, coordinate, seen=10)
    assert np.allclose(lines[0].coordinate, [0])
    assert np.allclose(lines[1].coordinate, [1, 0])
    assert np.allclose(time, [10, 1.0, 2.0])

    # Test addition order (limited by coordinate count)
    lines = [KymoLine([0], [0]), KymoLine([0], [1]), KymoLine([0], [2])]
    score = np.array([[2.0, 1.0], [4.0, 0.5], [10, 0]])
    time = [0.0, 1.0]
    coordinate = [0.0, 1.0]
    lines, time = assign_to_lines(score, lines, time, coordinate, seen=10)
    assert np.allclose(lines[2].coordinate, [2, 0])
    assert np.allclose(time, [10, 1])
