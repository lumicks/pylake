import numpy as np
import matplotlib as mpl


class ColorAdjustment:
    """Color adjustment for plotting"""

    def __init__(self, minimum, maximum, mode="absolute", gamma=1):
        """Utility class to adjust the min/max values of image colormaps.

        Values can be supplied as a single number or 1-component list, in which case the value is
        applied to all color channels, or a 3-component list specifying the values for red, green,
        and blue channels.

        Parameters
        ----------
        minimum : array_like
            1 or 3-component list of lower limits for the color mapping
        maximum : array_like
            1 or 3-component list of upper limits for the color mapping
        mode : str
            - "percentile" : color limits are given as percentiles of the image in question.
              Percentiles are calculated for each color channel separately
            - "absolute" : color limits are given as absolute values.

            Note: When providing bounds in percentiles, limits will change depending on which image
            you are looking at. When scrolling through a stack of images, the limits will not remain
            constant.
        gamma : float
            1 or 3-component list of gamma adjustments.

            Applies a power law to the data according to: `((data - vmin) / (vmax - vmin)) ** gamma`

        Examples
        --------
        ::

            from lumicks import pylake
            file = pylake.File("example.h5")

            # Plot scan with red color mapped from 5th to 95th percentile.
            adjustment = lk.ColorAdjustment(5, 95, mode="percentile")
            file.scans["scan"].plot(channel="red", adjustment=adjustment)

            # Plot scan with RGB colors mapped from 5th to 95th percentile.
            adjustment = lk.ColorAdjustment([5, 5, 5], [95, 95, 95], mode="percentile")
            file.scans["scan"].plot(adjustment=adjustment)

            stack = lk.CorrelatedStack("camera_recording.tiff")
            # Plot force 1x with this stack and adjust the contrast by specifying an absolute upper
            # and lower bound for the intensities.
            absolute_adjustment = lk.ColorAdjustment([50, 30, 20], [1000, 195, 95])
            stack.plot_correlated(file.force1x, channel="rgb", adjustment=absolute_adjustment)
        """
        minimum, maximum, gamma = (np.atleast_1d(x) for x in (minimum, maximum, gamma))
        self.mode = None if mode == "nothing" else mode
        if not self.mode:
            return

        for bound in (minimum, maximum, gamma):
            if len(bound) not in (1, 3):
                raise ValueError("Color value bounds and gamma should be of length 1 or 3")

        if mode not in ("percentile", "absolute"):
            raise ValueError("Mode must be percentile or absolute")

        self.minimum, self.maximum = minimum * np.ones((3,)), maximum * np.ones((3,))
        self.gamma = gamma * np.ones((3,))

    def _get_data_rgb(self, image):
        """Scale RGB data for plotting.

        Parameters
        ----------
        image : array_like
            Raw image data.
        """
        if not self.mode:
            return image / np.max(image)
        elif self.mode == "absolute":
            minimum, maximum = self.minimum, self.maximum
        else:
            bounds = np.array(
                [
                    np.percentile(img, [mini, maxi])
                    for img, mini, maxi in zip(np.moveaxis(image, 2, 0), self.minimum, self.maximum)
                ]
            )
            minimum, maximum = bounds.T

        denominator = maximum - minimum
        denominator[denominator == 0] = 1.0  # prevent div by zero
        return np.clip((image - minimum) / denominator, 0.0, 1.0) ** self.gamma

    def _update_limits(self, image_handle, image, channel):
        """Update color limits on an image generated by :func:`matplotlib.pyplot.imshow`

        Parameters
        ----------
        image_handle : matplotlib.image.AxesImage
            Image handle to apply color limits to
        image : array_like
            Raw image data.
        channel : str
            Channel that's being plotted (e.g. "red" or "rgb").
        """
        if not self.mode:
            return

        idx = {"red": 0, "green": 1, "blue": 2, "rgb": 0}[channel]
        limits = (self.minimum[idx], self.maximum[idx])
        if self.mode == "percentile":
            limits = np.percentile(image, limits)

        image_handle.set_norm(
            norm=mpl.colors.PowerNorm(vmin=limits[0], vmax=limits[1], gamma=self.gamma[idx])
        )

    @classmethod
    def nothing(cls):
        return cls(None, None, "nothing")
