import numpy as np
import os
import re
import json
import cv2
import tifffile
import warnings


class TiffFrame:
    """Thin wrapper around a TIFF frame stack. For camera videos timestamps are stored in the DateTime tag in
    the format start:end.

    Parameters
    ----------
    page : tifffile.tifffile.TiffPage
        Tiff page recorded from a camera in Bluelake.
    """
    def __init__(self, page, align):
        self._src = page
        self._align = align

    def _align_image(self):
        """ reconstruct image using alignment matrices from Bluelake; return aligned image as a NumPy array"""

        def correct_alignment_offset(alignment, x_offset, y_offset):
            # translate the origin of the image so that it matches that of the original transform
            translation = np.eye(3)
            translation[0, -1] = -x_offset
            translation[1, -1] = -y_offset
            # apply the original transform to the translated image. 
            # it only needs to be resized from a 2x3 to a 3x3 matrix
            original = np.vstack((alignment, [0, 0, 1]))
            # translate the image back to the original origin. 
            # takes into account both the offset and the scaling performed by the first step
            back_translation = np.eye(3)
            back_translation[0, -1] = original[0, 0] * x_offset
            back_translation[1, -1] = original[1, 1] * y_offset
            # concatenate the transforms by multiplying their matrices and ignore the unused 3rd row
            return np.dot(back_translation, np.dot(original, translation))[:2, :]

        try:
            description = json.loads(self._src.description)
        except json.decoder.JSONDecodeError:
            warnings.warn("File does not contain metadata. Only raw data is available")
            return self.raw_data

        try:
            align_mats = [np.array(description[f"Alignment {color} channel"]).reshape((2, 3))
                          for color in ('red', 'blue')]
            align_roi = np.array(description["Alignment region of interest (x, y, width, height)"])[:2]
            roi = np.array(description["Region of interest (x, y, width, height)"])[:2]
        except KeyError:
            warnings.warn("File does not contain alignment matrices. Only raw data is available")
            return self.raw_data
        
        x_offset, y_offset = align_roi - roi
        if not (x_offset == 0 and y_offset == 0):
            align_mats = [correct_alignment_offset(mat, x_offset, y_offset) for mat in align_mats]

        img = self.raw_data
        rows, cols, _ = img.shape
        for mat, channel in zip(align_mats, (0, 2)):
            img[:, :, channel] = cv2.warpAffine(img[:, :, channel], mat, (cols, rows),
                                                flags=(cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP),
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img

    @property
    def data(self):
        return self._align_image() if (self.is_rgb and self._align) else self._src.asarray()

    @property
    def raw_data(self):
        return self._src.asarray()

    @property
    def bit_depth(self):        
        bit_depth = self._src.tags["BitsPerSample"].value
        if self.is_rgb: # (int r, int g, int b)
            return bit_depth[0]
        else: # int
            return bit_depth

    @property
    def is_rgb(self):
        return self._src.tags["SamplesPerPixel"].value == 3

    def _get_plot_data(self, channel="rgb", vmax=None): 
        """ return data an numpy array, appropriate for use by `imshow`
            if data is grayscale or channel in ('red', 'green', 'blue')
                return data as is
            if channel is 'rgb',  converted to float in range [0,1] and correct for optional vmax argument:
                None  : normalize data to max signal of all channels
                float : normalize data to vmax value
        """
        
        if not self.is_rgb:
            return self.data

        if channel.lower() == "rgb":
            data = (self.data / (2**self.bit_depth - 1)).astype(np.float)
            if vmax is None:
                return data/data.max()
            else:
                return data/vmax
        else:
            try:
                return self.data[:, :, ("red", "green", "blue").index(channel.lower())]
            except ValueError:
                raise ValueError(f"'{channel}' is not a recognized channel")

    @property
    def start(self):
        timestamp_string = re.search(r'^(\d+):\d+$', self._src.tags['DateTime'].value)
        return np.int64(timestamp_string.group(1)) if timestamp_string else None

    @property
    def stop(self):
        timestamp_string = re.search(r'^\d+:(\d+)$', self._src.tags['DateTime'].value)
        return np.int64(timestamp_string.group(1)) if timestamp_string else None


class TiffStack:
    """TIFF images exported from Bluelake

    Parameters
    ----------
    tiff_file : tifffile.TiffFile
        TIFF file recorded from a camera in Bluelake.
    """
    def __init__(self, tiff_file, align):
        self._src = tiff_file
        self._align = align

    def get_frame(self, frame):
        return TiffFrame(self._src.pages[frame], align=self._align)

    @staticmethod
    def from_file(image_file, align):
        return TiffStack(tifffile.TiffFile(image_file), align=align)

    @property
    def num_frames(self):
        return len(self._src.pages)


class CorrelatedStack:
    """CorrelatedStack acquired with Bluelake. Bluelake can export stacks of images to various formats. These can be
    opened and correlated to timeline data using CorrelatedStack.

    Parameters
    ----------
    image_name : str
        Filename for the image stack. Typically a TIFF file recorded from a camera in Bluelake.

    align : bool
        If enabled, multi-channel images will be reconstructed from the image alignment metadata
        from Bluelake. The default value is `True`.

    Examples
    --------
    ::

        from lumicks import pylake

        # Loading a stack.
        stack = pylake.CorrelatedStack("example.tiff")

        # Making a plot where force is correlated to images in the stack.
        file = pylake.File("example.h5")
        stack.plot_correlated(file.force1x)

        # Determine the force trace averaged over frame 2...9.
        file.force1x.downsampled_over(stack[2:10].timestamps)
    """
    def __init__(self, image_name, align=True):
        self.src = TiffStack.from_file(image_name, align=align)
        self.name = os.path.splitext(os.path.basename(image_name))[0]
        self.start_idx = 0
        self.stop_idx = self.src.num_frames

    def __getitem__(self, item):
        """All indexing is in frames"""
        if isinstance(item, slice):
            if item.step is not None:
                raise IndexError("Slice steps are not supported")

            start, stop, _ = item.indices(self.num_frames)
            return CorrelatedStack.from_data(self.src, self.name, self.start_idx + start, self.start_idx + stop)
        else:
            item = self.start_idx + item if item >= 0 else self.stop_idx + item
            if item >= self.stop_idx or item < self.start_idx:
                raise IndexError("Index out of bounds")
            return CorrelatedStack.from_data(self.src, self.name, item, item + 1)

    def __iter__(self):
        idx = 0
        while idx < self.num_frames:
            yield self._get_frame(idx)
            idx += 1

    @classmethod
    def from_data(cls, data, name=None, start_idx=0, stop_idx=None):
        """Construct CorrelatedStack from image stack object

        Parameters
        ----------
        data : TiffStack
            TiffStack object.
        name : str
            Plot label of the correlated stack
        start_idx : int
            Index at the first frame.
        stop_idx: int
            Index beyond the last frame.
        """
        new_correlated_stack = cls.__new__(cls)
        new_correlated_stack.src = data
        new_correlated_stack.name = name
        new_correlated_stack.start_idx = start_idx
        new_correlated_stack.stop_idx = (new_correlated_stack.src.num_frames if stop_idx is None else stop_idx)
        return new_correlated_stack

    def plot(self, frame=0, channel="rgb", show_title=True, **kwargs):
        """Plot image from image stack

        Parameters
        ----------
        frame : int, optional
            Index of the frame to plot.
        channel : 'rgb', 'red', 'green', 'blue', None; optional
            Channel to plot for RGB images (None defaults to 'rgb')
            Not used for grayscale images
        show_title : bool, optional
            Controls display of auto-generated plot title
        **kwargs
            Forwarded to :func:`matplotlib.pyplot.imshow`.
        """
        import matplotlib.pyplot as plt

        default_kwargs = dict(
            cmap='gray',
            vmax=None
        )
        kwargs = {**default_kwargs, **kwargs}

        image = self._get_frame(frame)._get_plot_data(channel, vmax=kwargs["vmax"])
        plt.imshow(image, **kwargs)

        if show_title:
            if self.num_frames == 1:
                plt.title(self.name)
            else:
                # display with 1-based index for frames and total frames
                plt.title(f"{self.name} [frame {frame+1}/{self.num_frames}]")

    def _get_frame(self, frame=0):
        if frame >= self.num_frames or frame < 0:
            raise IndexError("Frame index out of range")
        return self.src.get_frame(self.start_idx + frame)

    def plot_correlated(self, channel_slice, frame=0, reduce=np.mean, channel="rgb"):
        """Downsample channel on a frame by frame basis and plot the results. The downsampling function (e.g. np.mean)
        is evaluated for the time between a start and end time of a frame. Note: In environments which support
        interactive figures (e.g. jupyter notebook with ipywidgets or interactive python) this plot will be interactive.

        Parameters
        ----------
        channel_slice : pylake.channel.Slice
            Data slice that we with to downsample.
        frame : int
            Frame to show.
        reduce : callable
            The function which is going to reduce multiple samples into one. The default is
            :func:`numpy.mean`, but :func:`numpy.sum` could also be appropriate for some cases
            e.g. photon counts.
        channel : 'rgb', 'red', 'green', 'blue', None; optional
            Channel to plot for RGB images (None defaults to 'rgb')
            Not used for grayscale images


        Examples
        --------
        ::

            from lumicks import pylake

            file = pylake.File("example.h5")
            stack = pylake.CorrelatedStack("example.tiff")
            stack.plot_correlated(file.force1x, frame=5)
        """
        import matplotlib.pyplot as plt

        downsampled = channel_slice.downsampled_over(self.timestamps, where='left', reduce=reduce)

        if len(downsampled.timestamps) < len(self.timestamps):
            warnings.warn("Only subset of time range available for selected channel")

        fetched_frame = self._get_frame(frame)
        aspect_ratio = fetched_frame.data.shape[0] / np.max([fetched_frame.data.shape])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plt.figaspect(aspect_ratio/2))
        t0 = downsampled.timestamps[0]
        t, y = (downsampled.timestamps - t0)/1e9, downsampled.data
        ax1.step(t, y, where='pre')
        ax2.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        image_object = ax2.imshow(fetched_frame._get_plot_data(channel=channel), cmap='gray')
        plt.title(f"Frame {frame}")

        # Make sure the y-axis limits stay fixed when we add our little indicator rectangle
        y1, y2 = ax1.get_ylim()
        ax1.set_ylim(y1, y2)

        def update_position(new_frame):
            return ax1.fill_between((np.array([new_frame.start, new_frame.stop]) - t0)/1e9, y1, y2, alpha=0.7, color='r')

        poly = update_position(fetched_frame)

        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel(downsampled.labels['y'])
        ax1.set_title(downsampled.labels['title'])
        ax1.set_xlim([np.min(t), np.max(t)])

        def select_frame(event):
            nonlocal poly

            if not event.canvas.widgetlock.locked() and event.inaxes == ax1:
                time = event.xdata * 1e9 + t0
                for img_idx in np.arange(0, self.num_frames):
                    current_frame = self._get_frame(img_idx)

                    if current_frame.start <= time < current_frame.stop:
                        plt.title(f"Frame {img_idx}")
                        poly.remove()
                        image_object.set_data(current_frame._get_plot_data(channel=channel))
                        poly = update_position(current_frame)
                        fig.canvas.draw()
                        return

        fig.canvas.mpl_connect('button_press_event', select_frame)

    @property
    def num_frames(self):
        """Number of frames in the stack."""
        return self.stop_idx - self.start_idx

    @property
    def raw(self):
        """Raw frame data."""
        if self.num_frames > 1:
            return [self._get_frame(idx) for idx in range(self.num_frames)]
        else:
            return self._get_frame(0)

    @property
    def start(self):
        """Starting time stamp of the stack."""
        return self._get_frame(0).start

    @property
    def stop(self):
        """Final time stamp of the stack."""
        return self._get_frame(self.num_frames - 1).stop

    @property
    def timestamps(self):
        """List of time stamps."""
        return [(self._get_frame(idx).start, self._get_frame(idx).stop) for idx in range(self.num_frames)]
