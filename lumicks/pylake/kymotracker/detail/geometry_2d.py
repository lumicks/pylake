from scipy.ndimage import gaussian_filter
from .linalg_2d import eigenvalues_2d_symmetric, eigenvector_2d_symmetric
import numpy as np


def find_subpixel_location(gx, gy, largest_eigenvalue, nx, ny):
    """This function determines the subpixel location at which the second derivative attains its minimum.

    It works by performing a Taylor expansion in the direction perpendicular to the line.
    and then analytically solving for the first derivative. In 1D:
      f(x) = r + r'x + r''x^2 / 2
      f'(x) = r' + r''x = 0
    ==> x = - r'/r''

    Parameters
    ----------
    gx: array_like
        First order Gaussian derivative in x direction.
    gy: array_like
        First order Gaussian derivative in y direction.
    largest_eigenvalue: array_like
        Second order Gaussian derivative in the direction perpendicular to the line (largest eigenvalue of the Hessian).
    nx: array_like
        Image containing x component of the vector normal to the line.
    ny: array_like
        Image containing y component of the vector normal to the line.

    Returns
    -------
    px: array_like
        Image containing the x coordinate of the subpixel location of the ridge.
    py: array_like
        Image containing the y coordinate of the subpixel location of the ridge.
    image: array_like
        Mask whether the optimum is inside or outside of the pixel.
    """

    # Evaluate the subpixel location of the line.
    # x = r'/r'', note that the largest eigenvalue corresponds to the directional derivative in the direction
    # perpendicular to the line (i.e. gxx * nx * nx + 2.0 * gxy * nx * ny + gyy * ny * ny)
    t = -(gx * nx + gy * ny) / largest_eigenvalue

    px = t * nx
    py = t * ny

    # Find points where the subpixel fit is inside
    inside = np.zeros(px.shape)
    x_condition = np.logical_and(px >= -0.5, px <= 0.5)
    y_condition = np.logical_and(py >= -0.5, py <= 0.5)
    inside[np.logical_and(x_condition, y_condition)] = 1

    return px, py, inside


def largest_second_derivative_2d(gxx, gxy, gyy):
    """This function determines the largest eigenvalue and corresponding eigen vector of an image based on Gaussian
    derivatives. It returns the direction and magnitude of the largest second derivative. For lines, this direction
    will correspond to the normal vector.

    Parameters
    ----------
    gxx: array_like
        Second order Gaussian derivative in x direction.
    gyy: array_like
        Second order Gaussian derivative in y direction.
    gxy: array_like
        Gaussian derivative w.r.t. x and y."""
    eigenvalues = eigenvalues_2d_symmetric(gxx, gxy, gyy)

    # Largest negative eigenvalue always seems to be the first eigenvalue, but evaluate just to be sure.
    max_eig = np.expand_dims(np.argmax(np.abs(eigenvalues), axis=2), axis=2)

    # Line strength (second derivative along steepest principal axis) is given by:
    # S = nx * nx * gxx + 2 * nx * ny * gxy + ny * ny * gyy
    # This is the same as the largest eigenvalue which we already computed.
    largest_eigenvalue = np.squeeze(np.take_along_axis(eigenvalues, max_eig, axis=2))

    # Normal perpendicular to the line
    nx, ny = eigenvector_2d_symmetric(gxx, gxy, gyy, largest_eigenvalue)

    return nx, ny, largest_eigenvalue


def calculate_image_geometry(data, sig_x, sig_y):
    """This function determines the largest eigenvalue and corresponding eigenvector of an image
    based on Gaussian derivatives. For sufficiently high sigma this corresponds to the line
    strength and normal to the line.

    For bar shaped objects, sigma needs to be above width/sqrt(3) otherwise they may not be
    detected as they have no well defined optimum.

    Parameters
    ----------
    data: array_like
        Input image.
    sig_x: float
        Standard deviation of the kernel. This needs to be chosen appropriate to detect objects
        at the correct scale (see function description).
    sig_y: float
        Standard deviation of the kernel. This needs to be chosen appropriate to detect objects
        at the correct scale (see function description).

    Returns
    -------
    largest_eig: np_array
        2D image containing largest eigenvalue of the Hessian.
    normals: np_array
        N by N by 2 array containing the normals of the image.
    positions: np_array
        N by N by 2 array containing subpixel coordinates of the maxima.
    inside: np_array
        2D image mask whether it is a potential line center or not.
    """
    gx = gaussian_filter(data, [sig_x, sig_y], order=[1, 0])
    gy = gaussian_filter(data, [sig_x, sig_y], order=[0, 1])
    gxx = gaussian_filter(data, [sig_x, sig_y], order=[2, 0])
    gyy = gaussian_filter(data, [sig_x, sig_y], order=[0, 2])
    gxy = gaussian_filter(data, [sig_x, sig_y], order=[1, 1])

    nx, ny, largest_eig = largest_second_derivative_2d(gxx, gxy, gyy)
    px, py, inside = find_subpixel_location(gx, gy, largest_eig, nx, ny)

    normals = np.stack((nx, ny), axis=2)
    positions = np.stack((px, py), axis=2)

    return largest_eig, normals, positions, inside


def is_opposite(trial_normals, reference):
    """This function checks whether the smallest angle between the vectors in trial_normals and reference are smaller
    than 0.5 pi; if not, that means a smaller angle is possible by flipping them.

    We do this by checking the dot products between the current normal and each of the trial points.

    dot(a,b) = |a| |b| cos(theta), where theta is the minimum angle between the vectors.

    Any minimum angular distance larger than .5 pi can be made smaller by flipping the normal. We don't explicitly have
    to calculate the arccos as we know that this valid range maps from [cos(0), cos(.5 pi)] to [1, 0].

    In other words, any dot product smaller than zero needs its normal flipped."""
    dot_normal = np.dot(trial_normals, reference)
    return dot_normal < 0


def is_in_2d(coord, shape):
    # TO DO: Find a better way.
    return (
        (coord[:, 0] >= 0)
        & (coord[:, 0] < shape[0])
        & (coord[:, 1] >= 0)
        & (coord[:, 1] < shape[1])
    )


def get_candidate_generator():
    """This function returns a candidate generator. This candidate generator needs to be called with a normal angle and
    produces the relative coordinate it is pointing to and the closest two adjacent points.

    Examples:
       X X O      X X X
    →  X o O   ↘ X o O
       X X O      X O O
    """

    # Given a discretized angle of a normal vector, this lookup table returns the pixel the vector
    # perpendicular to that the normal is pointing at and its adjacent pixels closest to the origin.
    # It assumes the angle is discretized as np.round(4.0/np.pi*angle) + 4, mapping from[-pi, pi] to [0, 8].
    # Index 0 corresponds to the normal pointing to -pi. This means the line is pointing to .5 * pi or [0 -1].
    candidate_lut = np.array(
        [
            [[1, -1], [0, -1], [-1, -1]],  # - Pi
            [[1, 0], [1, -1], [0, -1]],
            [[1, -1], [1, 0], [1, 1]],
            [[0, 1], [1, 1], [1, 0]],
            [[-1, 1], [0, 1], [1, 1]],
            [[-1, 0], [-1, 1], [0, 1]],
            [[-1, -1], [-1, 0], [-1, 1]],
            [[0, -1], [-1, -1], [-1, 0]],
            [[1, -1], [0, -1], [-1, -1]],  # Pi
        ],
        dtype=int,
    )

    def generate_candidates(angle):
        return candidate_lut[int(np.round(4 * angle / np.pi) + 4)]

    return generate_candidates
