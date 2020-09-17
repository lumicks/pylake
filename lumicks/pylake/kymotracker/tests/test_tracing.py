from lumicks.pylake.kymotracker.detail.scoring_functions import build_score_matrix, kymo_score
from lumicks.pylake.kymotracker.detail.trace_line_2d import KymoLine, append_next_point, extend_line, \
    points_to_line_segments
from lumicks.pylake.kymotracker.detail.peakfinding import KymoPeaks
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
    matrix = np.reshape(build_score_matrix(lines, times, positions, kymo_score(vel=0, sigma=.5, diffusion=.125),
                                           sigma_cutoff=2), ((len(unique_times), -1)))
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
    matrix = np.reshape(build_score_matrix(lines, times, positions, kymo_score(vel=1, sigma=.5, diffusion=.125),
                                           sigma_cutoff=2), (len(unique_times), -1))
    reference = [
        [-np.inf, -np.inf, -np.inf, -1.0, -0.0, -1.0, -np.inf],
        [-np.inf, -np.inf, -np.inf, -2.745166004060959, -0.6862915010152397, -0.0, -0.6862915010152397],
        [-np.inf, -np.inf, -np.inf, -np.inf, -2.1435935394489816, -0.5358983848622454, -0.0],
        [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -1.7777777777777777, -0.4444444444444444],
        [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -3.4376941012509463, -1.5278640450004204],
        [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -3.0254695407844476],
    ]
    assert np.allclose(matrix, reference)


def test_line_append():
    kymoline = KymoLine([0.0], [2.0])
    frame = KymoPeaks.Frame(np.array([1.0, 2.0, 2.5]), np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0]))
    frame.reset_assignment()

    def score_fun(line, time, coord):
        score = -np.abs(line[0].coordinate[0]-coord)
        score[-1] = -np.inf
        return score

    assert append_next_point(kymoline, frame, score_fun)
    assert np.allclose(kymoline.time, [0.0, 1.0])
    assert np.allclose(kymoline.coordinate, [2.0, 2.0])
    assert np.array_equal(frame.unassigned, [True, False, True])

    # Coordinate 3 is closer, but last score returns -np.inf for score
    assert append_next_point(kymoline, frame, score_fun)
    assert np.allclose(kymoline.time, [0.0, 1.0, 1.0])
    assert np.allclose(kymoline.coordinate, [2.0, 2.0, 1.0])
    assert np.array_equal(frame.unassigned, [False, False, True])

    # Only coordinate 3 is left, but returns -np.inf and should not be considered a candidate ever. Terminate line!
    assert not append_next_point(kymoline, frame, score_fun)  # No more assignable coordinates
    assert np.allclose(kymoline.time, [0.0, 1.0, 1.0])
    assert np.allclose(kymoline.coordinate, [2.0, 2.0, 1.0])
    assert np.array_equal(frame.unassigned, [False, False, True])


def test_extend_line():
    peaks = KymoPeaks(
        np.array([1.0, 2.0, 3.0, 4.0, 6.0, 5.0, 7.0]),
        np.array([1.0, 2.0, 3.0, 1.0, 3.0, 2.0, 5.0]),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    )
    peaks.reset_assignment()

    def score_fun(line, time, coord):
        score = -np.abs(line[0].coordinate[-1] - coord)
        return score

    # Starting from 4 we should get the extension 4, 5, 6 first. 7 will not be included, since it is at time point 5
    # and our window only goes up one frame
    kymoline = KymoLine([0.0], [4.0])
    extend_line(kymoline, peaks, 1, score_fun)
    assert np.allclose(kymoline.coordinate, np.array([4.0, 4.0, 5.0, 6.0]))

    # 4, 5, 6 no longer being available, we should get 1, 2, 3
    kymoline = KymoLine([0.0], [4.0])
    extend_line(kymoline, peaks, 1, score_fun)
    assert np.allclose(kymoline.coordinate, np.array([4.0, 1.0, 2.0, 3.0]))

    # With a bigger window, we should get 7.0 too
    peaks.reset_assignment()
    kymoline = KymoLine([0.0], [4.0])
    extend_line(kymoline, peaks, 2, score_fun)
    assert np.allclose(kymoline.coordinate, np.array([4.0, 4.0, 5.0, 6.0, 7.0]))

    # Starting from t=2, we should only get 6
    # and our window only goes up one frame
    peaks.reset_assignment()
    kymoline = KymoLine([2.0], [5.0])
    extend_line(kymoline, peaks, 1, score_fun)
    assert np.allclose(kymoline.coordinate, np.array([5.0, 6.0]))


def test_kymotracker_two_integration():
    peaks = KymoPeaks(
        np.array([1.0, 2.0, 3.0, 4.0, 6.0, 5.0, 7.0]),
        np.array([1.0, 2.0, 3.0, 1.0, 3.0, 2.0, 5.0]),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    )

    lines = points_to_line_segments(peaks, kymo_score(vel=0, sigma=1, diffusion=0), window=8, sigma_cutoff=2)
    assert np.allclose(lines[0].coordinate, [1.0, 2.0, 3.0])
    assert np.allclose(lines[1].time, [1.0, 2.0, 3.0, 5.0])
    assert np.allclose(lines[1].coordinate, [4.0, 5.0, 6.0, 7.0])

    lines = points_to_line_segments(peaks, kymo_score(vel=0, sigma=1, diffusion=0), window=1, sigma_cutoff=2)
    assert np.allclose(lines[0].coordinate, [1.0, 2.0, 3.0])
    assert np.allclose(lines[1].time, [1.0, 2.0, 3.0])
    assert np.allclose(lines[1].coordinate, [4.0, 5.0, 6.0])
    assert np.allclose(lines[2].time, [5.0])
    assert np.allclose(lines[2].coordinate, [7.0])


def test_sampling():
    test_img = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ])

    # Tests the bound handling
    kymoline = KymoLine([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    assert np.allclose(kymoline.sample_from_image(test_img, 50), [0, 2, 3, 2, 0])
    assert np.allclose(kymoline.sample_from_image(test_img, 2), [0, 2, 3, 2, 0])
    assert np.allclose(kymoline.sample_from_image(test_img, 1), [0, 2, 2, 2, 0])
    assert np.allclose(kymoline.sample_from_image(test_img, 0), [0, 1, 1, 1, 0])
    assert np.allclose(KymoLine([0, 1, 2, 3, 4], [4, 4, 4, 4, 4]).sample_from_image(test_img, 0), [0, 0, 1, 1, 0])

    kymoline = KymoLine([0.1, 1.1, 2.1, 3.1, 4.1], [0.1, 1.1, 2.1, 3.1, 4.1])
    assert np.allclose(kymoline.sample_from_image(test_img, 50), [0, 2, 3, 2, 0])
    assert np.allclose(kymoline.sample_from_image(test_img, 2), [0, 2, 3, 2, 0])
    assert np.allclose(kymoline.sample_from_image(test_img, 1), [0, 2, 2, 2, 0])
    assert np.allclose(kymoline.sample_from_image(test_img, 0), [0, 1, 1, 1, 0])
    assert np.allclose(KymoLine([0.1, 1.1, 2.1, 3.1, 4.1], [4.1, 4.1, 4.1, 4.1, 4.1]).sample_from_image(test_img, 0),
                       [0, 0, 1, 1, 0])
