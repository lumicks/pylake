import numpy as np
import pytest

from lumicks.pylake.detail import widefield


def make_image_gaussians(spots, sigma=15, amplitude=1):
    sigma = np.eye(2) * sigma
    x_coord, y_coord = np.meshgrid(np.arange(0, 100), np.arange(0, 100))
    shape = x_coord.shape
    coordinates = np.vstack((x_coord.ravel(), y_coord.ravel()))

    image = np.zeros(shape)
    for x, y in spots:
        mu = np.array([x, y])[:, np.newaxis]
        coord_diff = coordinates - mu
        quad_form = np.sum(np.dot(coord_diff.T, np.linalg.inv(sigma)) * coord_diff.T, axis=1)
        image += amplitude * np.exp(-0.5 * quad_form).reshape(shape)
    return image


def test_transform_default():
    transform = widefield.TransformMatrix()
    np.testing.assert_equal(transform.matrix, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))


def test_transform_mul_type():
    with pytest.raises(
        TypeError, match="Operands must be of type `TransformMatrix`, got <class 'int'>."
    ):
        widefield.TransformMatrix() * 1


def test_transform_inversion():
    theta = np.radians(20)
    m = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    transform = widefield.TransformMatrix(m[:2])
    np.testing.assert_allclose(transform.invert().matrix, np.linalg.inv(m))


# fmt: off
@pytest.mark.parametrize("theta_deg, center, ref_matrix", [
    (45, (5, 5), [[0.70710678, 0.70710678, -2.07106781], [-0.70710678, 0.70710678, 5.0], [0, 0, 1]]),
    (45, (5, 6), [[0.70710678, 0.70710678, -2.77817459], [-0.70710678, 0.70710678, 5.29289322], [0, 0, 1]]),
    (30, (5, 6), [[0.8660254, 0.5, -2.33012702], [-0.5, 0.8660254, 3.30384758], [0, 0, 1]]),

])
# fmt: on
def test_transform_rotation(theta_deg, center, ref_matrix):
    theta = np.radians(theta_deg)
    rotation = widefield.TransformMatrix.rotation(theta_deg, center)

    # test matrix is calculated appropriately from angle, center parameters
    np.testing.assert_allclose(rotation.matrix, np.array(ref_matrix), rtol=1e-7)

    def rotate_point(x, y):
        new_x = (x - center[0]) * np.cos(theta) - (y - center[1]) * np.sin(theta) + center[0]
        new_y = (x - center[0]) * np.sin(theta) + (y - center[1]) * np.cos(theta) + center[1]
        return np.array((new_x, new_y))

    # test coordinate rotation
    # rotate points with trig functions, undo the rotation with TransformMatrix
    original_points = [np.array((0, 5)), center, np.array((10, 5))]
    rot_points = [rotate_point(*p) for p in original_points]

    coordinates = rotation.warp_coordinates(rot_points)
    np.testing.assert_allclose(coordinates[1], center)  # rotation center
    np.testing.assert_allclose(coordinates, original_points, atol=1e-8)

    coordinates = rotation.invert().warp_coordinates(original_points)
    np.testing.assert_allclose(coordinates[1], center)  # rotation center
    np.testing.assert_allclose(coordinates, rot_points, atol=1e-8)


@pytest.mark.parametrize("theta, center", [(25, (25, 50)), (45, (25, 30))])
def test_rotate_image(theta, center):
    rotation = widefield.TransformMatrix.rotation(theta, np.array(center))
    rotation_wrong_angle = widefield.TransformMatrix.rotation(theta + 0.01, np.array(center))
    rotation_wrong_origin = widefield.TransformMatrix.rotation(
        theta, np.array((center[0] + 0.05, 50))
    )

    original_spots = [(25, 50), (75, 50)]
    original_image = make_image_gaussians(original_spots)

    target_spots = rotation.warp_coordinates(original_spots)
    target_image = make_image_gaussians(target_spots)

    # absolute tolerance set to allow for interpolation error
    # 2% error from a peak amplitude of 1
    rotated_image = rotation.warp_image(original_image)
    np.testing.assert_allclose(rotated_image, target_image, atol=0.02)

    # test that transformation using the wrong matrix fails
    assert np.max(np.abs(rotation_wrong_angle.warp_image(original_image), target_image)) > 0.02
    assert np.max(np.abs(rotation_wrong_origin.warp_image(original_image), target_image)) > 0.02


@pytest.mark.parametrize("x, y", [(0, 0), (1, 1), (-3, 4), (2, -6)])
def test_transform_translation(x, y):
    points = ((1, 2), (3, 4), (6, 5), (-5, -5))
    translation = widefield.TransformMatrix.translation(x, y)

    expected_points = np.vstack(points) + (x, y)
    new_points = translation.warp_coordinates(points)
    np.testing.assert_allclose(np.vstack(new_points), expected_points)


@pytest.mark.parametrize(
    "theta1, theta2, center",
    [(5, 10, (8, 6)), (-5, 10, (8, 6)), (0, 0, (0, 0)), (0, 10, (6, 8)), (5, 0, (6, 8))],
)
def test_transform_multiplication(theta1, theta2, center):
    theta1 = np.radians(theta1)
    theta2 = np.radians(theta2)
    center = np.array(center)

    make_mat = lambda t, x, y: np.array(
        [
            [np.cos(t), -np.sin(t), x],
            [np.sin(t), np.cos(t), y],
            [0, 0, 1],
        ]
    )

    a = make_mat(theta1, *center)
    b = make_mat(theta2, *center)
    c = np.matmul(b, a)

    mat_a = widefield.TransformMatrix(a[:2])
    mat_b = widefield.TransformMatrix(b[:2])
    mat_c = mat_b * mat_a

    np.testing.assert_allclose(mat_c.matrix, c)


def test_transform_multiplication_identity():
    mat_a = widefield.TransformMatrix.rotation(10, (5, 10))
    mat_b = widefield.TransformMatrix(np.array([[1, 0, 0], [0, 1, 0]]))
    mat_c = widefield.TransformMatrix()
    mat_d = widefield.TransformMatrix.rotation(0, (0, 0))

    np.testing.assert_equal(mat_a.matrix, (mat_b * mat_a).matrix)
    np.testing.assert_equal(mat_a.matrix, (mat_c * mat_a).matrix)
    np.testing.assert_equal(mat_a.matrix, (mat_d * mat_a).matrix)


def test_tether():
    origin = (0, 0)
    point_1 = np.array((1.46446609, 1.46446609))
    point_2 = np.array((8.53553391, 8.53553391))

    # empty tether
    tether = widefield.Tether((0, 0), None)
    assert tether._ends is None
    np.testing.assert_allclose(tether.rot_matrix.matrix, widefield.TransformMatrix().matrix)
    assert tether.ends is None

    # test coordinates of tether after rotation
    tether = widefield.Tether(origin, (point_1, point_2))
    np.testing.assert_allclose(tether.ends[0], (0, 5), atol=1e-8)
    np.testing.assert_allclose(tether.ends[1], (10, 5))

    # test offsets
    tether = widefield.Tether((1, 2), (point_1 - (1, 2), point_2 - (1, 2)))
    np.testing.assert_allclose(tether.ends, [(-1, 3), (9, 3)])

    # test re-define offsets
    tether = widefield.Tether((0, 0), (point_1, point_2))
    tether = tether.with_new_offsets((1, 2))
    np.testing.assert_allclose(tether.ends, [(-1, 3), (9, 3)])
