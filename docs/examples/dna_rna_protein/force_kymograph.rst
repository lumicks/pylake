Kymograph and Force
===================

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`


We have two beads trapped and a DNA attached to both of them at either end. We made sure that we have a single tether of DNA by pulling on them before and doing the FD curve.

We then moved into the channel that contains Sytox-Green. It binds to DNA if the DNA is under tension. We can the scan along the DNA and create kymographs using the confocal part of the system.

As we start the kymographs, we can change the force on the DNA and observe the force dependent binding of Sytox to DNA.

This experiments perfectly demonstrates the correlative capabilities of the C-trap.

Open the file::

    # Sytox binding, unbinding, with decreased, than increased force
    file = lk.File("20181107-152940 Sytox kymograph 7.h5")

Make Kymographs
---------------

List all the kymographs in the file::

    >>> list(file.kymos)
    ['7']

Load the kymograph in the file::

    # you can either do this and then you have to change which kymo you load for every file:
    kymo_data = file.kymos["7"] # as this file contains kymograph #7

    # ALTERNATIVELY you can either do this and then you don't have to worry about which file you open
    kymo_names = list(file.kymos)
    kymo = file.kymos[kymo_names[0]]

Plot the green channel::

    plt.figure(figsize=(15, 10))
    kymo.plot("green")

.. image:: force_kymograph1.png

Note that we can also scale the colorbar of the image.

This is not so straightforward, here we just show a very simple way of doing it.

Get the raw data out of the kymographs::

    blue_date = kymo.blue_image
    green_date = kymo.green_image
    red_date = kymo.red_image

    # this gives you the timestamps if you want to produce the kymos yourself
    timestamps = kymo.timestamps

Get a sense of the pixel values in the kymos

    >>> max_px = np.max(green_date)
    35
    >>> min_px = np.min(green_date)
    0

Scale the colorbar and make the kymograph look better::

    plt.figure(figsize=(15,10))
    kymo.plot("green", vmax=10)

.. image:: force_kymograph2.png

Force versus Time
-----------------

Load the data::

    # Force in the x direction (pN)
    forcex = file['Force HF']['Force 1x']

    # time traces (seconds)
    time = forcex.timestamps/1e9
    time = time - time[0]

    sample_rate = forcex.sample_rate

Downsample the data::

    downsampled_rate = 100 # Hz

    # downsample the force, nanostage position and time
    forcex_downsamp = forcex.downsampled_by(int(sample_rate/downsampled_rate))
    time_downsamp = forcex_downsamp.timestamps/1e9
    time_downsamp = time_downsamp - time_downsamp[0]

Plot the force::

    plt.figure(figsize=(10,5))

    forcex.plot(label="Original")
    forcex_downsamp.plot(color='r',label="Downsampled")
    plt.ylabel('Force 1x (pN)')
    plt.xlim([0,max(time)])
    plt.legend()

.. image:: force_kymograph3.png


Correlated Force and Confocal
-----------------------------

Plot the final figure::

    plt.figure(figsize=(15,10))

    plt.subplot(2,1,1)
    kymo.plot("green", vmax=10)

    plt.subplot(2,1,2)
    forcex.plot(label="Original")
    forcex_downsamp.plot(color='r',label="Downsampled")
    plt.xlim([0, max(time)])
    plt.ylabel('Force 1x (pN)')

.. image:: force_kymograph4.png

We see when we decreased the force on the DNA the Sytox unbound. As soon as we increase the tension back, we see Sytox binding again. At around 52 seconds, the DNA tether broke, which is why the force went back to it's original position.
