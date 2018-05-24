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

file = lumicks.hdf5.File("example.h5")

# Plot all kymographs in a file
for name, kymo in file.kymos.items():
    kymo.plot_rgb()
    plt.savefig(name)
    
# Simple force plotting
file.force1x.plot()
plt.savefig("force1x")

# Accessing the raw data
f1x_data = file.force1x.data
f1x_timestamps = file.force1x.timestamps
plt.plot(f1x_timestamps, f1x_data)
```
