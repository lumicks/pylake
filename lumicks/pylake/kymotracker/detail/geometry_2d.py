from scipy.ndimage import gaussian_filter
from .linalg_2d import eigenvalues_2d_symmetric, eigenvector_2d_symmetric
import numpy as np


def find_subpixel_location(gx, gy, gxx, gxy, gyy, nx, ny):
    """This function determines the subpixel location at which the second derivative attains its minimum.
    It works by performing a Taylor expansion in the direction of the gradient (perpendicular to the line)
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
    gxx: array_like
        Second order Gaussian derivative in x direction.
    gyy: array_like
        Second order Gaussian derivative in y direction.
    gxy: array_like
        Gaussian derivative w.r.t. x and y.
    nx: array_like
        Image containing x component of the vector normal to the line.
    ny: array_like
        Image containing y component of the vector normal to the line.

    Returns
    -------
    px: array_like
        Image containing the x coordinate of the subpixel location of the ridge.
    px: array_like
        Image containing the x coordinate of the subpixel location of the ridge.
    image: array_like
        Mask whether the optimum is inside or outside of the pixel.
    """

    # Evaluate the subpixel location of the line
    denominator = (gxx * nx * nx + 2.0 * gxy * nx * ny + gyy * ny * ny)
    t = - (gx * nx + gy * ny) / denominator
    px = t * nx
    py = t * ny

    # Find points where the subpixel fit is inside
    inside = np.zeros(px.shape)
    x_condition = np.logical_and(px >= -.5, px <= .5)
    y_condition = np.logical_and(py >= -.5, py <= .5)
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

    # Largest absolute eigenvalue always seems to be the first eigenvalue, but evaluate just to be sure.
    max_eig = np.expand_dims(np.argmax(np.abs(eigenvalues), axis=2), axis=2)

    # Line strength (second derivative along gradient) is given by:
    # S = nx * nx * gxx + 2 * nx * ny * gxy + ny * ny * gyy
    # This is the same as the largest eigenvalue which we already computed.
    largest_eigenvalue = np.squeeze(np.take_along_axis(eigenvalues, max_eig, axis=2))

    # Normal perpendicular to the line
    nx, ny = eigenvector_2d_symmetric(gxx, gxy, gyy, largest_eigenvalue)

    assert np.allclose(nx * nx * gxx + 2 * nx * ny * gxy + ny * ny * gyy, largest_eigenvalue)

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
        2D image containing largest eigenvalue of the Hessian
    normals: np_array
        N by N by 2 array containing the normals of the image
    positions: np_array
        N by N by 2 array containing subpixel coordinates of the maxima
    inside: np_array
        2D image mask whether it is a line point or not.
    """
    gx = gaussian_filter(data, [sig_x, sig_y], order=[1, 0])
    gy = gaussian_filter(data, [sig_x, sig_y], order=[0, 1])
    gxx = gaussian_filter(data, [sig_x, sig_y], order=[2, 0])
    gyy = gaussian_filter(data, [sig_x, sig_y], order=[0, 2])
    gxy = gaussian_filter(data, [sig_x, sig_y], order=[1, 1])

    nx, ny, largest_eig = largest_second_derivative_2d(gxx, gxy, gyy)
    px, py, inside = find_subpixel_location(gx, gy, gxx, gxy, gyy, nx, ny)

    normals = np.stack((nx, ny), axis=2)
    positions = np.stack((px, py), axis=2)

    return largest_eig, normals, positions, inside
