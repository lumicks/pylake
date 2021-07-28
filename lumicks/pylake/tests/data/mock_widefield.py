import numpy as np
import json
import tifffile


class MockTag():
    def __init__(self, value):
        self._value = value

    @property
    def value(self):
        return self._value


class MockTiffPage:
    def __init__(self, data, start_time, end_time, description="", bit_depth=8):
        self._data = data
        bit_depth = bit_depth if data.ndim == 2 else (bit_depth, bit_depth, bit_depth)
        self.tags = {"DateTime": MockTag(f"{start_time}:{end_time}"),
                     "ImageDescription": MockTag(description),
                     "BitsPerSample": MockTag(bit_depth),
                     "SamplesPerPixel": MockTag(1 if (data.ndim==2) else data.shape[2])}

    def asarray(self):
        return self._data.copy()

    @property
    def description(self):
        return self.tags["ImageDescription"].value


class MockTiffFile:
    def __init__(self, data, times, description="", bit_depth=8):
        self.pages = []
        for d, r in zip(data, times):
            self.pages.append(MockTiffPage(d, r[0], r[1], description=description, bit_depth=bit_depth))

    @property
    def num_frames(self):
        return len(self._src.pages)


def make_alignment_image_data(spots, Tx_red, Ty_red, theta_red, Tx_blue, Ty_blue, theta_blue, bit_depth,
                              offsets=None, version=1):
    def make_transform_matrix(Tx, Ty, theta):
        M = np.eye(3)
        M[0, -1] = Tx
        M[1, -1] = Ty
        theta = np.radians(theta)
        M[0, 0] = np.cos(theta)
        M[0, 1] = -np.sin(theta)
        M[1, 0] = np.sin(theta)
        M[1, 1] = np.cos(theta)
        return M

    def transform_spots(M, spots):
        # apply channel offsets
        # reshape spots into coordinate matrix; [x,y,z] as columns
        N = spots.shape[1]
        spots = np.vstack((spots, np.ones(N)))
        return np.dot(M, spots)[:2]

    def make_image(spots_red, spots_green, spots_blue, bit_depth):
        # RGB image, 2D (normalized) gaussians at spot locations
        sigma = np.eye(2)*5
        X, Y = np.meshgrid(np.arange(0, 200), np.arange(0, 100))
        img = np.zeros((*X.shape, 3))
        for j, pts in enumerate((spots_red.T, spots_green.T, spots_blue.T)):
            for x, y, in pts:
                mu = np.array([x,y])[:,np.newaxis]
                XX = np.vstack((X.ravel(), Y.ravel())) - mu
                quad_form = np.sum(np.dot(XX.T, np.linalg.inv(sigma)) * XX.T, axis=1)
                Z = np.exp(-0.5 * quad_form)
                img[:, :, j] += Z.reshape(X.shape)
            img[:, :, j] = img[:, :, j] / img[:, :, j].max()
        return (img * (2**bit_depth - 1)).astype(f"uint{bit_depth}")

    def make_description(m_red, m_blue, offsets):
        if offsets is None:
            offsets = [0, 0]
        # WARP_INVERSE_MAP flag requires original transformation that resulted in un-aligned image
        if version == 1:
            labels = [f"Alignment {c} channel" for c in ("red", "green", "blue")]
        elif version == 2:
            labels = [f"Channel {x} alignment" for x in range(3)]
        return {labels[0]: m_red[:2].ravel().tolist(),
                labels[1]: np.eye(3)[:2].ravel().tolist(),
                labels[2]: m_blue[:2].ravel().tolist(),
                "Alignment region of interest (x, y, width, height)": [offsets[0], offsets[1], 200, 100],
                "Region of interest (x, y, width, height)": [0, 0, 200, 100]}

    spots = np.array(spots).T # [2 x N]
    img0 = make_image(spots, spots, spots, bit_depth)

    if offsets is not None: # translate origin by offsets
        spots[0] -= offsets[0]
        spots[1] -= offsets[1]
    m_red = make_transform_matrix(Tx_red, Ty_red, theta_red)
    m_blue = make_transform_matrix(Tx_blue, Ty_blue, theta_blue)
    red_spots = transform_spots(m_red, spots)
    blue_spots = transform_spots(m_blue, spots)
    if offsets is not None: # back-translate origin
        for tmp_spots in (red_spots, spots, blue_spots):
            tmp_spots[0] += offsets[0]
            tmp_spots[1] += offsets[1]

    img = make_image(red_spots, spots, blue_spots, bit_depth)
    description = make_description(m_red, m_blue, offsets)

    return img0, img, description


def write_tiff_file(img0, img, description, bit_depth, n_frames, filename):
    if bit_depth == 8:
        img = img[:,:,0]
        channels = 1
    else:
        channels = 3
    movie = np.stack([img for n in range(n_frames)], axis=0)

    tag_orientation = (274, 'H', 1, 1, False) # Orientation = ORIENTATION.TOPLEFT
    tag_sample_format = (339, 'H', channels, (1, )*channels, False) # SampleFormat = SAMPLEFORMAT.UINT

    with tifffile.TiffWriter(filename) as tif:
        for n, frame in enumerate(movie):
            str_datetime = f"{n*10+10}:{n*10+18}"
            tag_datetime = (306, 's', len(str_datetime), str_datetime, False)
            tif.save(frame,
                    description=json.dumps(description, indent=4),
                    software="Bluelake Unknown",
                    metadata=None, contiguous=False,
                    extratags=(tag_orientation, tag_sample_format, tag_datetime))
