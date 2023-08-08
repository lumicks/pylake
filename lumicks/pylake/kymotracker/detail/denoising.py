import numpy as np
import scipy


def equal_length(kernels):
    """Make the kernels the same length (while keeping them centered)

    Parameters
    ----------
    kernels : list of np.ndarray
        List of kernels to symmetrize.

    Raises
    ------
    ValueError
        if any of the kernel isn't odd sized (violates the ability to center it).
    """
    target_length = len(kernels[-1])

    def pad(kernel):
        if len(kernel) % 2 == 0:
            raise ValueError("This function should only be used with odd sized kernels")

        padding = np.zeros(((target_length - len(kernel)) // 2, 1))
        return np.vstack((padding, kernel, padding))

    return [pad(k) for k in kernels]


def generate_product_filters(kernels):
    """Generate product filters

    Generates a list of kernels that consist of the previous kernels convolved.

    Parameters
    ----------
    kernels : list of np.ndarray
        List of 1-dimensional kernels to calculate the convolutional product of.
    """
    current = np.asarray([[1.0]])
    convolved_kernels = [current]
    for k in kernels:
        current = scipy.signal.convolve(current, k, mode="full")
        convolved_kernels.append(current)

    return convolved_kernels


def generate_bspline_kernels(num_kernels):
    """Returns a list of à trous b3 spline kernels.

    Parameters
    ----------
    num_kernels : int
        Number of kernels we want

    Returns
    -------
    kernels : list of np.ndarray
        1-dimensional kernels of varying sizes
    """
    k = np.arange(1, num_kernels + 1)
    base_kernel = np.array([1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16])
    kernels = []
    n_kernels = len(base_kernel)
    for zeros in 2 ** (k - 1) - 1:
        kernel = np.zeros(n_kernels + zeros * (n_kernels - 1))
        kernel[:: zeros + 1] = base_kernel
        kernels.append(kernel.reshape((-1, 1)))

    return kernels


def tau(kernel, power):
    """Calculate filter summation from Sec 2. Paragraph 1."""
    return np.sum(kernel**power)


def calculate_vst_coefficients(kernel):
    """Calculate optimal transform coefficients and expected standard deviation.

    We calculate b and c for the following transform:

        Z = b sign(Y + c) * sqrt(Y + c)

    Using the coefficients `c` from Eq. 10 and `b` (which is defined above Eq. 12 in [3]_), we
    estimate a transform such that Z - b sqrt(tau_1 lambda) ~ N(0, 1) where lambda is the Poisson
    parameter [3]_.

    Parameters
    ----------
    kernel : np.ndarray
        Kernel at scale `j`. Note that this is the full kernel, so Kj * Kj-1 .. Kn
        where * is convolution operator.

    References
    ----------
    .. [3] Zhang, B., Fadili, J. M., & Starck, J. L. (2008). Wavelets, ridgelets, and curvelets
           for Poisson noise removal. IEEE Transactions on image processing, 17(7), 1093-1108.
    """
    b = np.sign(tau(kernel, 1)) / np.sqrt(np.abs(tau(kernel, 1)))  # Defined above eqn 12
    c = 7.0 * tau(kernel, 2) / (8.0 * tau(kernel, 1)) - tau(kernel, 3) / (2.0 * tau(kernel, 2))
    return b, c


def calculate_vst_stdev(kernel, prev_kernel):
    """Calculate expected standard deviation of detail layer

    Parameters
    ----------
    kernel, prev_kernel : np.ndarary
        kernel at scale j and j - 1

    Raises
    ------
    ValueError
        if kernel sizes are not equal
    """
    if prev_kernel.shape != kernel.shape:
        raise ValueError("Kernel shapes must be equal to calculate standard deviation")

    term1 = tau(prev_kernel, 2) / (4 * tau(prev_kernel, 1) ** 2)
    term2 = tau(kernel, 2) / (4 * tau(kernel, 1) ** 2)
    term3 = np.sum(prev_kernel * kernel) / (2 * tau(prev_kernel, 1) * tau(kernel, 1))
    expected_sd = np.sqrt(term1 + term2 - term3)

    return expected_sd


def variance_stabilizing_transform(image, coefficients):
    b, c = coefficients
    return b * np.sign(image + c) * np.sqrt(np.abs(image + c))


def inverse_variance_stabilizing_transform(image, coefficients):
    b, c = coefficients
    return np.sign(image) * (image / b) ** 2 - c


def determine_significant(image, stdev, false_detection_rate=0.1):
    """Determines significant pixels in an image

    This method assumes that pixels are distributed according to N(0, stdev) and tests which pixels
    have significant signal in them. Considering that we are doing a large number of tests, we
    apply a correction to control the number of false positives we get. This correction is given
    by the Benjamini-Hochberg procedure [1].

    Parameters
    ----------
    image : np.ndarray
        Image
    stdev : float
        Expected standard deviation of the distribution based on theoretical considerations.
    false_detection_rate : float
        Represents the False Detection Rate when it comes to pixels with significant
        signal. The probability of erroneously detecting spots in a spot-free homogeneous
        noise is upper bounded by this value. The FDR is technically given by:
        false_positive / (false_positive + true_positive).

    Raises
    ------
    ValueError
        if standard deviation is negative

    References
    ----------
    .. [1] Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical
           and powerful approach to multiple testing. Journal of the Royal statistical society:
           series B (Methodological), 57(1), 289-300.
    """
    if stdev < 0:
        raise ValueError("Standard deviation must be positive")

    raw_pvalues = 2.0 * (1.0 - scipy.stats.norm.cdf(np.abs(image) / stdev))
    pvalues = np.sort(raw_pvalues.flatten())

    # TODO: Offer a correction for dependent samples
    num_tests = len(pvalues)
    pvalue_threshold = false_detection_rate * np.arange(1, num_tests + 1) / num_tests
    comp = pvalues < pvalue_threshold
    p_cutoff = pvalues[comp][-1] if np.any(comp) else 0.0

    return raw_pvalues <= p_cutoff


class MultiScaleVarianceStabilizingTransform:
    """Calculates the required coefficients for determining the variance stabilizing transform (VST)

    The VST is a generalization of the Anscombe transform which can convert Poisson distributed.
    noise to approximately Gaussian distributed noise. For Poisson distributed data, the variance
    scales with the mean. After the transform the distribution is roughly Gaussian with a variance
    of 1. This transformation allows you to apply statistics that rely on normality post-transform.

    By combining this with a wavelet procedure, the variance and bias of this procedure goes down
    more quickly than when applied without any filters [1, 2].

    Parameters
    ----------
    kernels : list of np.ndarray
        List of kernels to use for the multiscale variance stabilizing transform.
    two_dimensional : bool
        Convert the kernel into a two-dimensional kernel.

    References
    ----------
    .. [1] Zhang, B., Fadili, M. J., Starck, J. L., & Olivo-Marin, J. C. (2007, September).
           Multiscale variance-stabilizing transform for mixed-Poisson-Gaussian processes and its
           applications in bioimaging. In 2007 IEEE International Conference on Image Processing
           (Vol. 6, pp. VI-233). IEEE.
    .. [2] Zhang, B., Fadili, J. M., & Starck, J. L. (2008). Wavelets, ridgelets, and curvelets
           for Poisson noise removal. IEEE Transactions on image processing, 17(7), 1093-1108.
    """

    def __init__(self, kernels, *, two_dimensional=False):
        self._kernels = [k * k.T for k in kernels] if two_dimensional else kernels

        # Full kernels contain h0 * h1 * ... * hn where * is convolution.
        equal_length_kernels = equal_length(generate_product_filters(kernels))
        self._full_kernels = [k * k.T if two_dimensional else k for k in equal_length_kernels]
        self._coefficients = [calculate_vst_coefficients(k) for k in self._full_kernels]

        self._stdev = [
            calculate_vst_stdev(k1, k2)
            for k1, k2 in zip(self._full_kernels[1:], self._full_kernels[:-1])
        ]

    def _calculate_wavelets(self, image, stabilize):
        """Calculate the wavelet coefficient images and remainder of a IUWT wavelet transform

        Parameters
        ----------
        image : np.ndarray
            Image to decompose into wavelet coefficients.
        stabilize : bool
            Applies a transformation that stabilizes the variance for the coefficient images.

        Returns
        -------
        detail_coefficients : List[np.ndarray]
            List of images representing the detail coefficients for each wavelet.
        remainder : np.ndarray
            Image containing the remainder (what's left after adding the detail layers).
        """
        filtered_imgs = [scipy.ndimage.convolve(image, kernel) for kernel in self._full_kernels]

        # Stabilize variance
        if stabilize:
            filtered_imgs = [
                variance_stabilizing_transform(img, coeffs)
                for img, coeffs in zip(filtered_imgs, self._coefficients)
            ]

        detail_coefficients = [f1 - f2 for f1, f2 in zip(filtered_imgs, filtered_imgs[1:])]
        remainder = filtered_imgs[-1]

        return detail_coefficients, remainder

    def _reconstruct_image(self, detail_coefficients, remainder, stabilize):
        """Reconstructs image from wavelet decomposition.

        Parameters
        ----------
        detail_coefficients : List[np.ndarray]
            List of images representing the detail coefficients for each wavelet.
        remainder : np.ndarray
            Image containing the remainder (what's left after adding the detail layers).
        stabilize : bool
            Indicate that the detail coefficients were stabilized and this transform
            must be inverted.
        """
        output_image = sum(detail_coefficients + [remainder])
        if not stabilize:
            return output_image

        return inverse_variance_stabilizing_transform(output_image, self._coefficients[0])

    def filter_image(self, image, false_detection_rate=0.1, verbose=False):
        """Filter an image with MS-VST.

        Parameters
        ----------
        image : np.ndarray
            Image array
        false_detection_rate : float
            Represents the False Detection Rate when it comes to pixels with significant
            signal. The probability of erroneously detecting spots in a spot-free homogeneous
            noise is upper bounded by this value.
        verbose : bool
            Show extra output
        """
        image = np.array(image, dtype=float)  # integer arrays lose quality when convolved
        detail_coefficients, remainder = self._calculate_wavelets(image, stabilize=True)

        if verbose:
            for d, stdev in zip(detail_coefficients, self._stdev):
                print(f"practical: {np.std(d.flatten())}, theoretical: {stdev}")

        # Determine significant coefficients. At this point, the distribution is approximately
        # normal and the standard deviation is known by construction.
        significant = [
            determine_significant(d, stdev, false_detection_rate=false_detection_rate)
            for d, stdev in zip(detail_coefficients, self._stdev)
        ]

        # Filter the insignificant coefficients
        detail_coefficients = [d * sig for d, sig in zip(detail_coefficients, significant)]

        # Inverse transform
        output_image = self._reconstruct_image(detail_coefficients, remainder, stabilize=True)

        # Enforce positivity. Error model states that negative values don't make sense. Yet they
        # can still occur because of the overcompleteness of the IUWT and the fact that we
        # modified values in the detail coefficients directly.
        output_image[output_image < 0] = 0

        return output_image, significant

    def filter_regularized(
        self,
        image,
        false_detection_rate=0.1,
        num_iter=10,
        remove_background=True,
        verbose=False,
    ):
        """Filter the image using a regularized wavelet reconstruction.

        This reconstructs the image, but with the additional constraint that the
        resulting image has to be positive and sparse (L1 regularization). This regularization
        procedure helps, since the IUWT is overcomplete.

        The regularization (shrinkage) is performed by shrinking coefficients towards zero. This
        is done via a soft-thresholding procedure where coefficients below a threshold are set to
        zero, while values above the threshold are shrunk towards zero. The threshold is then
        decreased each step (this is a requirement for the algorithm). Note that at each step, the
        significant structures are forced to be maintained.

        As a result, we will get a sparse representation that fulfills positivity, while
        keeping all the coefficients that were significant.

        Parameters
        ----------
        image : np.ndarray
            Image array
        false_detection_rate : float
            Represents the False Detection Rate when it comes to pixels with significant
            signal. The probability of erroneously detecting spots in a spot-free homogeneous
            noise is upper bounded by this value.
        num_iter : int
            Number of iterations to run.
        remove_background : bool
            Remove the background after approximating the image? This amounts to not adding
            the final approximation layer.
        verbose : bool
            Show some output while it is running.
        """
        image = np.array(image, dtype=float)  # integer arrays lose quality when convolved

        # Grab the significant wavelet coefficients first.
        _, significant = self.filter_image(image, false_detection_rate=false_detection_rate)

        detailed_coeffs, remainder = self._calculate_wavelets(image, stabilize=False)

        def positivity_projector(coefficients):
            """Calculate the coefficients that will lead to a fully positive reconstruction

            This corresponds to the operator Qs2 in the paper [1].

            Parameters
            ----------
            coefficients : list of np.ndarray
                Wavelet detail coefficients
            """
            img = sum(coefficients) + remainder
            positive_img = np.clip(img, 0, np.inf)
            coefficients, rem = self._calculate_wavelets(positive_img, stabilize=False)
            return coefficients

        def significance_enforcer(coefficients):
            """Force significant structures to remain.

            This corresponds to the operator Ps in the paper [1].

            Parameters
            ----------
            coefficients : list of np.ndarray
                Wavelet detail coefficients
            """
            for c, d, sig in zip(coefficients, detailed_coeffs, significant):
                c[sig] = d[sig]
            return coefficients

        current_coeffs = [np.copy(d) for d in detailed_coeffs]
        beta = 1.0

        for k in range(num_iter):
            positive = positivity_projector(current_coeffs)
            next_solution = significance_enforcer(positive)

            if verbose:
                print(f"Iter: {k}: Beta={beta}")

            def soft_threshold(detail_coeffs, treshold):
                """Soft-thresholding in the context of L1 optimization involves stepping towards
                zero, and truncating any values below the stepsize to zero"""
                detail_coeffs[abs(detail_coeffs) < treshold] = 0.0
                detail_coeffs[detail_coeffs > treshold] -= treshold
                detail_coeffs[detail_coeffs < -treshold] += treshold
                return detail_coeffs

            current_coeffs = [soft_threshold(d.copy(), beta) for d in next_solution]
            beta -= 1.0 / num_iter

        reconstructed_img = sum(current_coeffs)
        if not remove_background:
            reconstructed_img += remainder

        return np.clip(reconstructed_img, 0.0, np.inf)
