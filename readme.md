# Lumicks Python toolbox

This is a Python package with various tools for Bluelake.

## Install

For general use, all you need to do is enter the following on the command line:

```bash
pip install lumicks
```

To upgrade to the latest version:

```bash
pip install -U lumicks
```

## Reading HDF5 files

```python
import lumicks

h5file = lumicks.hdf5.File("example.h5")
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

### Scans and kymographs

The following code uses kymographs as an example. 
Scans work the same way -- just substitute `h5file.kymos` with `h5file.scans`.

```python
# Plot all kymographs in a file
for name, kymo in h5file.kymos.items():
    kymo.plot_rgb()
    plt.savefig(name)

# Pick a single FD curve
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
```
