Fitting a power spectrum
------------------------

In the previous section, the physical origin of the power spectrum was introduced.
However, there are some practical aspects to consider.
So far, we have only considered the expectation value of the power spectrum.
In reality, power spectral values follow a distribution.

The real and imaginary part of the frequency spectrum are normally distributed.
As a consequence, the squared magnitude of the power spectrum is exponentially distributed.
This has two consequences:

- Fitting the power spectral values directly using a simple least squares fitting routine, we would
  get very biased estimates. These estimates would overestimate the plateau and corner frequency,
  resulting in overestimated trap stiffness and force response and an underestimated distance response.
- The signal to noise ratio is poor (equal to one :cite:`norrelykke2010power`).

A commonly used method for dealing with this involves data averaging, which trades resolution for an
improved signal to noise ratio. In addition, by virtue of the central limit theorem, data averaging
leads to a more symmetric data distribution (more amenable to standard least-squares fitting procedures).

There are two ways to perform such averaging:

- The first is to split the time series into windows of equal length, compute the power spectrum for
  each chunk of data and averaging these. This procedure is referred to as *windowing*.
- The second is to calculate the spectrum for the full dataset followed by downsampling in the
  spectral domain by averaging adjacent bins according to :cite:`berg2004power`. This procedure is
  referred to as *blocking*.

We use the blocking method for spectral averaging, since this allows us to reject noise peaks at high
resolution prior to averaging. Note however, that the error incurred by this blocking procedure depends
on :math:`n_b`, the number of points per block, :math:`\Delta f`, the spectral resolution and inversely
on the corner frequency :cite:`berg2004power`.

Setting the number of points per block too low would result in a bias from insufficient averaging
:cite:`berg2004power`. Insufficient averaging would result in an overestimation of the force response
(:math:`R_f`) and an underestimation of the distance response (:math:`R_d`). In practice, one should
use a high number of points per block (:math:`n_b \gg 100`), unless a very low corner frequency precludes this.
In such cases, it is preferable to increase the measurement time.
