import numpy as np


def eigenvalues_2d_symmetric(a, b, d):
    """This function returns the eigenvalues of a 2x2 symmetric matrix given by:
    | a  b |
    | b  d |

    Note that this a special case of a 2x2 symmetric matrix where every element of the matrix is passed as an image.
    This allows the evaluation of eigenvalues to be vectorized over the entire image. This is much more efficient
    than calling the numpy functions for computing the eigenvalues for each pixel of the image.

    Parameters
    ----------
    a, b, d: array_like
        2D images containing the values of the matrix elements a b and d.
    """
    t = a + d
    bsq4 = 4 * b ** 2
    amdsq = (a - d) ** 2
    ht = .5 * t
    half_sqrt_discriminant = .5 * np.sqrt(bsq4 + amdsq)
    eig1 = ht + half_sqrt_discriminant
    eig2 = ht - half_sqrt_discriminant

    return np.stack((eig1, eig2), axis=len(eig1.shape))


def eigenvector_2d_symmetric(a, b, d, eig, eps=1e-8):
    """Returns normalized eigenvector corresponding to the provided eigenvalue.

    Note that this a special case of a 2x2 symmetric matrix where every element of the matrix is passed as an image.
    This allows the evaluation of eigenvalues to be vectorized over the entire image. This is much more efficient
    than calling the numpy function for computing the eigenvectors for each pixel of the image.

    This function solves:
    | a-lambda  b        |
    | b         d-lambda | [x, y] = 0

    Which means that:
      bx = (lambda - d) y
    or
      y = (lambda - a)/b x

    This solution is invalid for b == 0. Here we expect orthogonal vectors [1 0] and [0 1].
        ax + by = l x
        bx + dy = l y

    so x = 1 iff b = 0 and l = a
    and y = 1 iff b = 0 and l = d
    """
    ex = np.zeros(a.shape)
    ey = np.zeros(a.shape)
    ex[np.abs(a - eig) < eps] = 1
    ey[np.abs(d - eig) < eps] = 1

    mask = np.abs(b) > eps
    tx = b[mask]
    ty = eig[mask] - a[mask]

    length = np.sqrt(tx * tx + ty * ty)
    tx = tx / length
    ty = ty / length

    ex[mask] = tx
    ey[mask] = ty

    return ex, ey
