Files and channels
==================

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

Opening a Bluelake HDF5 file is very simple::

    from lumicks import pylake

    file = pylake.File("example.h5")

Contents
--------

To see a textual representation of the contents of a file::

    >>> print(file)
    File root metadata:
    - Bluelake version: 1.3.1
    - Experiment: Example
    - Description: Collecting example data for Pylake
    - GUID: {1A8024D2-C49B-48FF-B183-2FDF0065F26D}
    - Export time (ns): 1531162366497820300
    - File format version: 1

    Force HF:
      Force 1x:
      - Data type: float64
      - Size: 706251
      Force 1y:
      - Data type: float64
      - Size: 706251
      Force 2x:
      - Data type: float64
      - Size: 706251
      Force 2y:
      - Data type: float64
      - Size: 706251
    Info wave:
      Info wave:
      - Data type: uint8
      - Size: 706251
    Photon count:
      Blue:
      - Data type: uint32
      - Size: 706251
      Green:
      - Data type: uint32
      - Size: 706251
      Red:
      - Data type: uint32
      - Size: 706251

    .scans
      - reference
      - bleach
      - imaging

    .markers:
      - FRAP 3

    .force1x
      .calibration
    .force1y
      .calibration
    .force2x
      .calibration
    .force2y
      .calibration

For a listing of more specific timeline items::

    >> list(file.fdcurves)
    ['baseline', '1', '2']

    >>> list(file.scans)
    ['reference', 'bleach', 'imaging']

    >>> list(file.kymos)
    ['5', '6', '7']

They can also be printed to get more information::

    >>> print(file.scans)
    {'reference': Scan(pixels=(67, 195)),
     'bleach': Scan(pixels=(20, 19)),
     'imaging': Scan(pixels=(20, 19))}


Channels
--------

Just like the Bluelake timeline, exported HDF5 files contain multiple channels of data.
They can be easily accessed as shown below::

    file.force1x.plot()
    plt.savefig("force1x.png")

The channels have a few convenient methods, like `.plot()` which make it easy to preview the contents, but you can also always access the raw data directly::

    f1x_data = file.force1x.data
    f1x_timestamps = file.force1x.timestamps
    plt.plot(f1x_timestamps, f1x_data)

The above examples use the `force1x` channel.
A full list of available channels can be found on the :class:`~lumicks.pylake.File` reference page.

By default, entire channels are returned from a file::

    everything = file.force1x
    everything.plot()

But channels can easily be sliced::

    # Get the data between 1 and 1.5 seconds
    part = file.force1x['1s':'1.5s']
    part.plot()
    # Or manually
    f1x_data = part.data
    f1x_timestamps = part.timestamps
    plt.plot(f1x_timestamps, f1x_data)

    # More slicing examples
    a = file.force1x[:'-5s']  # everything except the last 5 seconds
    b = file.force1x['-1m':]  # take the last minute
    c = file.force1x['-1m':'-500ms']  # last minute except the last 0.5 seconds
    d = file.force1x['1.2s':'-4s']  # between 1.2 seconds and 4 seconds from the end
    e = file.force1x['5.7m':'1h 40m']  # 5.7 minutes to an hour and 40 minutes

    # Subslicing is also possible
    a = file.force1x['1s':]  # from 1 second to the end of the file
    b = a['1s':]  # 1 second relative to the start of slice `a`
                  # --> `b` starts at 2 seconds relative to the beginning of the file

Note that channels are indexed in time units using numbers with suffixes.
The possible suffixes are d, h, m, s, ms, us, ns, corresponding to day, hour, minute, second, millisecond, microsecond and nanosecond.
This indexing only applies to channels slices.
Once you access the raw data, those are regular arrays which use regular array indexing::

    channel_slice = file.force1x['1.5s':'20s']  # timestamps
    data_slice = file.force1x.data[20:40]  # indices into the array

Calibrations
------------

Calibration information for force channels can be found by checking the calibration member. This gives a list of calibrations::

    >>> print(file.force1x.calibration)
    [{'Kind': 'Discard all calibration data', 'Offset (pN)': 0.0, 'Response (pN/V)': 1.0, 'Sign': 1.0, 'Start time (ns)': 0, 'Stop time (ns)': 0}]

The actual values can be obtained from the list as follows, where the index refers to the calibration entry and the name to the actual field value::

    >>> file.force1x.calibration[0]["Offset (pN)"]
    0.0

If we slice a force channel, we only obtain the calibrations relevant for the selected region.

Markers
-------

We can see that the file also contains markers. These can be accessed from the markers attribute which returns a dictionary of markers.

    >>> print(file.markers)
    {'FRAP 3': <lumicks.pylake.marker.Marker at 0x2c6164bc910>}

The actual markers can be obtained from the dictionary as follows::

    >>> file.markers["FRAP 3"]
    <lumicks.pylake.marker.Marker at 0x2c616bcf8b0>

We can find the start and stop time with ``.start`` and ``.stop``.

    >>> print(file.markers["FRAP 3"].start)
    1573136459289265920

    >>> print(file.markers["FRAP 3"].stop)
    1573136602571107585
