# Lumicks pylake 

[![DOI](https://zenodo.org/badge/133832492.svg)](https://zenodo.org/badge/latestdoi/133832492)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](license.md)
![Build Status](https://github.com/lumicks/pylake/workflows/pytest/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/lumicks-pylake/badge/?version=latest)](https://lumicks-pylake.readthedocs.io/en/latest/?badge=latest)

This Python package includes data analysis tools for Bluelake HDF5 data.

## Install

For general use, all you need to do is enter the following on the command line:

```bash
pip install lumicks.pylake
```

To upgrade to the latest version:

```bash
pip install -U lumicks.pylake
```

## Reading HDF5 files

```python
from lumicks import pylake

h5file = pylake.File("example.h5")
```

### FD curves

```python
import matplotlib.pyplot as plt

# Plot all FD curves in a file
for name, fd in h5file.fdcurves.items():
    fd.plot_scatter()
    plt.savefig(name)

# Pick a single FD curve
fd = h5file.fdcurves["name"]
# By default, the FD channel pair is `downsampled_force2` and `distance1`
fd.with_channels(force='1x', distance='2').plot_scatter()

# Access the raw data: defaults
force = fd.f
distance = fd.d
# Access the raw data: specific
force = fd.downsampled_force1y
distance = fd.distance2

# Plot manually: FD curve
plt.scatter(distance.data, force.data)
# Plot manually: force timetrace
plt.plot(force.timestamps, force.data)

# By default `f` is `downsampled_force2` and `d` is `distance1`
altenative_fd = fd.with_channels(force='1x', distance='2')

# Baseline subtraction
fd_baseline = h5file.fdcurves["Baseline"]
fd_measured = h5file.fdcurves["Measurement"]
fd = fd_measured - fd_baseline
fd.plot_scatter()
```

### Force vs. time

```python
# Simple force plotting
h5file.force1x.plot()
plt.savefig("force1x")

# Accessing the raw data
f1x_data = h5file.force1x.data
f1x_timestamps = h5file.force1x.timestamps
plt.plot(f1x_timestamps, f1x_data)
```

### Slicing data channels

```python
# Take the entire channel
everything = h5file.force1x
everything.plot()

# Get the data between 1 and 1.5 seconds
part = h5file.force1x['1s':'1.5s']
part.plot()
# Or manually
f1x_data = part.data
f1x_timestamps = part.timestamps
plt.plot(f1x_timestamps, f1x_data)

# More slicing examples
a = h5file.force1x[:'-5s']  # everything except the last 5 seconds
b = h5file.force1x['-1m':]  # take the last minute
c = h5file.force1x['-1m':'-500ms']  # last minute except the last 0.5 seconds
d = h5file.force1x['1.2s':'-4s']  # between 1.2 seconds and 4 seconds from the end
e = h5file.force1x['5.7m':'1h 40m']  # 5.7 minutes to an hour and 40 minutes
```

### Scans and kymographs

The following code uses kymographs as an example. 
Scans work the same way -- just substitute `h5file.kymos` with `h5file.scans`.

```python
# Plot all kymographs in a file
for name, kymo in h5file.kymos.items():
    kymo.plot_rgb()
    plt.savefig(name)

# Pick a single kymograph
kymo = h5file.kymos["name"]
# Plot a single color channel
kymo.plot_red()

# Access the raw image data
rgb = kymo.rgb_image  # matrix
blue = kymo.blue_image
# Plot manually
plt.imshow(rgb)

# Low-level raw data
photons = kymo.red_photons
plt.plot(photons.timestamps, photons.data)

# Saving photon counts to TIFF
kymo.save_tiff("kymograph.tiff")
```

```python
scan = h5file.scans["name"]

# A scan can have multiple frames
print(scan.num_frames)
print(scan.blue_image.shape)  # (self.num_frames, h, w) -> single color channel
print(scan.rgb_image.shape)  # (self.num_frames, h, w, 3) -> three color channels
```
