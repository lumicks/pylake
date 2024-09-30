Near surface calibration example
================================

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

In this notebook, we will look at calibration data acquired at different distances from the surface.
In this example, we aim to show three things.

- How to get an estimate of the height above the flowcell surface.
- Why active calibration is preferred for surface assays.
- How to analyze active calibration data.

This data will be used to assess the effect the nearby surface has on the estimated calibration factors.
We will be using both passive and active calibration and compare the obtained results.
First we download the data::

    lk.download_from_doi(r"10.5281/zenodo.13880274", "surface_calibration")

We will use the `glob <https://docs.python.org/3/library/glob.html#glob.glob>`_ module to collect
file names according to a pattern conveniently.
During acquisition, we have stored the height in the file name to make it easier to identify the data.
We can extract this height from the filename using the regular expression module `re <https://docs.python.org/3/library/re.html>`_.
Let's import both these modules now::

    import re
    import glob

To get a list of files, we use glob::

    filenames = glob.glob(r"surface_calibration/*Marker*.h5")

The experimental condition for each calibration is stored in its filename.
We write two small helper files, that allow us to extract these experimental conditions conveniently::

    def height(filename):
        """Grabs the approximate height from the filename"""

        # Searches for a string of the format value.value, where the decimal point part is optional
        return float(re.search(r"\d+[.]\d+", filename)[0])

    def osc_axis(filename):
        """Grabs the oscillation axis from the filename"""

        # Searches for nanostage_X or nanostage_Y and extracts the axis by grabbing the last.
        # character of a match. If no match is found, then it must be a passive calibration.
        matches = re.search(r"nanostage_([XY])", filename)
        return matches[0][-1] if matches else "passive"

To be able to compare passive and active calibration, we need to have a reasonable estimate of the height above the flow cell surface.
To approximately determine this height, we approach the surface until the axial force steeply rises.
At that point, we know that we have hit the surface with the bead.
We then calculate the surface position based on the inflection point of a piecewise linear fit.
With the surface position at hand, we can use the stage position to then calculate the height above the cover-slip surface.

There is one more issue to consider however.
When moving the nanostage, one might assume that moving the nanostage up by `1` micron, would result
in a trap height that is `1` micron less than before.
This is not the case however.

The trap height is determined by the position of the stage, but also by the focal shift that occurs
when focusing through interfaces with mismatched indices of refraction (such as water and glass).
This focal shift introduces a scaling factor between the vertical motion of the stage surface and
the axial position of the trap within the flowcell.

For paraxial rays, this focal shift can be computed from Snell’s law, but it is not straightforward
to compute when high NA objectives are used :cite:`neuman2004optical`.

There is a way to measure this shift for surface experiments.
When trapping near the surface, the light reflected between the bead and the cover-glass gives rise
to an interference pattern.
In other words, there is a spatial modulation of the intensity as a function of the axial position of the bead.
This is depicted schematically below (note that none of the depicted elements or angles are to scale).

.. image:: figures/interference.png
  :nbattach:

The spatial frequency of this intensity modulation is given by :cite:`neuman2004optical,neuman2005measurement`:

.. math::

    f_{spatial} = \frac{2 n_{medium}}{\lambda}

Where :math:`f_{spatial}` is the spatial intensity modulation we would obtain in the absence
of focal shift, :math:`\lambda` the laser wavelength and :math:`n_{medium}` the index of
refraction of the medium.

To find the focal shift, we measure the interference pattern on the light intensity by moving the
stage axially and measuring the axial force.
We then fit a sine-wave with an exponential decay term added to a polynomial background to this data:

.. math::

    I(z) = P_{background}(z) + \exp{\left(- k z\right)} \sin\left(2 \pi f_{observed} z\right)

Here :math:`I(z)` refers to the light intensity with :math:`z` the axial position,
:math:`P_{background}(z)` to the average light intensity as a function of axial position,
:math:`k` the decay constant and `f_{observed}` the observed frequency of the intensity modulation.
The effective focal shift is given by :math:`f_{observed} / f_{spatial}`.
Note that this relation is only valid over a small range, as further away, the change in axial
stiffness may also move the bead relative to the trap focus.

To perform this analysis in Pylake, we can use the function :func:`~lumicks.pylake.touchdown()`.
Let's load a file and perform the analysis::

    f = lk.File("surface_calibration/20220203-165705_Touchdown_T1_Fz.h5")
    td = lk.touchdown(
        f["Nanostage position"]["Z"].data[:len(f.force1z.data)],
        f.force1z.data,
        int(f.force1z.sample_rate)
    )

We can plot the result to inspect the quality of the fit and the point identified as the point where the bead touches the surface::

    plt.figure()
    td.plot()

.. image:: figures/touchdown.png

This fit looks reasonable, but we have to remember that these quantities are approximations.
The focal shift in reality is not a constant and the intersection point between the two linear
regressions of the axial force a crude approximation.

To obtain the distance of the bead above the cover-slip, we can use the determined focal shift
and the stage position at which the bead and the surface touched.
The relationship between these two is given by:

.. math::

    d / 2 - \alpha_{shift} \left(z_{nanostage} - z_{surface}\right)

Here :math:`d` is the bead diameter, :math:`\alpha_{shift}` is the focal shift factor,
:math:`z_{nanostage}` is the nanostage position and :math:`z_{surface}` is the nanostage position
at which the bead and flowcell touch (surface-to-surface).

To obtain `z_surface` and the focal shift we can use the properties
:attr:`~lumicks.pylake.Touchdown.surface_position` and
:attr:`~lumicks.pylake.Touchdown.focal_shift`.
Let's see what value we got for the focal shift.

    >>> td.focal_shift
    0.9131828139774159

These measurements were done with a water objective.
The value we obtain is close to `1`, which is what we would expect for a water objective.
Generally, for a water immersion objective, we'd expect values between `0.9` and `1.05`, whereas
a TIRF or oil objective would have focal shift values between `0.75` and `0.85`.

In this case, we do not need to use those values directly as a function to calculate
the height above the surface that we require for calibration directly is also available
as :meth:`~lumicks.pylake.calculate_height()`.

Given that we now have a way to calculate the height, let's create a small function to perform
the active and passive calibrations.
This helps keep the rest of our code short (rather than repeating the same parameters many times).
The function will take the force signal, a fit range (since we should use a different fit range
for axial force, `z`, than lateral force), the height above the surface,
and the nanostage data (for active calibrations)::

    # These variables will be picked up by the function as well.
    bead_diameter = 1.32

    # Note that we oscillated at 38 Hz, most C-Traps will oscillate at 17 Hz
    oscillation_frequency = 38

    def calibrate(force_signal, fit_range, nano=None, height=None):
        # Decalibrate the data back to volts by dividing by the old force response
        voltage = force_signal / force_signal.calibration[0]["Response (pN/V)"]

        calibration = lk.calibrate_force(
            voltage.data,
            driving_data=nano.data if nano else None,
            bead_diameter=bead_diameter,
            temperature=25,
            sample_rate=voltage.sample_rate,
            active_calibration=True if nano else False,
            driving_frequency_guess=oscillation_frequency,
            num_points_per_block=250,
            distance_to_surface=height,
            hydrodynamically_correct=False,  # We are too close to the surface to use hydro
            fit_range=fit_range,
        )

        return calibration

Let's define a function to do all the calibrations corresponding to a list of files.
To make comparisons on the effect of including the height determination in the calibration clear,
we add a parameter that defines whether we should be using the height information or not.
That way, we can see the effect of this on both passive and active calibration::

    def calibrate_files(filenames, use_height):
        calibrations = {}

        for filename in filenames:
            # Load the file and extract our channels of interest
            fh = lk.File(filename)
            f1x = fh.force1x
            f1y = fh.force1y
            f1z = fh.force1z
            n1x = fh["Nanostage position"]["X"]
            n1y = fh["Nanostage position"]["Y"]

            # We store our data by the height estimate present in the file name.
            # Have we encountered this height before? If not, add it to the dictionary!
            key = height(filename)
            if key not in calibrations:
                calibrations[key] = {}

            # We grab the oscillation axis from the file name (this will return "X", "Y" or "passive").
            oscillation_axis = osc_axis(filename)

            # We will store the average nanostage z-position for later use
            z_position = np.mean(fh["Nanostage position"]["Z"].data)
            calibrations[key]["Z"] = z_position

            # We calculate the height based on the touchdown data. We use this for the calibration.
            if use_height:
                current_height = td.calculate_height(z_position, bead_diameter)
            else:
                current_height = None

            calibrations[key]["current_height"] = current_height

            if oscillation_axis == "X":
                calibrations[key]["ac_x"] = calibrate(f1x, [100, 17000], n1x, height=current_height)
            elif oscillation_axis == "Y":
                calibrations[key]["ac_y"] = calibrate(f1y, [100, 17000], n1y, height=current_height)
            else:
                calibrations[key]["pc_x"] = calibrate(f1x, [100, 17000], height=current_height)
                calibrations[key]["pc_y"] = calibrate(f1y, [100, 17000], height=current_height)

                # Note that axial force needs a more limited fitting range!
                calibrations[key]["pc_z"] = calibrate(f1z, [60, 6000], height=current_height)

        return calibrations

We can now perform the calibrations.
First we do them while taking into account the (approximate) height above the surface::

    calibrations = calibrate_files(filenames, True)

Now it's time to see what we got::

    # Let's grab all the approximate heights (the keys of our dictionary) and sort them
    keys = np.sort(list(calibrations.keys()))

    # Let's also grab the heights we inferred from our stage position
    heights = np.asarray([calibrations[k]["current_height"] for k in keys])

We will plot the resulting stiffness values obtained using the various calibration methods::

    plt.figure()
    plt.plot(heights, [calibrations[h]["ac_x"].stiffness for h in keys], label="active X")
    plt.plot(heights, [calibrations[h]["ac_y"].stiffness for h in keys], label="active Y")

    plt.plot(heights, [calibrations[h]["pc_x"].stiffness for h in keys], 'C0--', label="passive X")
    plt.plot(heights, [calibrations[h]["pc_y"].stiffness for h in keys], 'C1--', label="passive Y")
    plt.plot(heights, [calibrations[h]["pc_z"].stiffness for h in keys], 'C2--', label="passive Z")
    plt.legend(loc="upper right")
    plt.title("Active vs passive calibration with approximate height")
    plt.xlabel('Height [um]')
    plt.ylabel('Stiffness [pN/nm]');

As we can see, the difference between passive and active calibration is not so large.
The stiffness is almost constant for all methods as we approach the surface.

.. image:: figures/ac_pc_with_height.png

What would have happened if we did not know the height above the surface?
Let's rerun the calibrations without the height information to check::

    calibrations_no_height = calibrate_files(filenames, False)

And plotting the stiffness again::

    plt.figure()
    plt.plot(heights, [calibrations_no_height[h]["ac_x"].stiffness for h in keys], label="active X")
    plt.plot(heights, [calibrations_no_height[h]["ac_y"].stiffness for h in keys], label="active Y")

    plt.plot(heights, [calibrations_no_height[h]["pc_x"].stiffness for h in keys], 'C0--', label="passive X")
    plt.plot(heights, [calibrations_no_height[h]["pc_y"].stiffness for h in keys], 'C1--', label="passive Y")
    plt.plot(heights, [calibrations_no_height[h]["pc_z"].stiffness for h in keys], 'C2--', label="passive Z")
    plt.legend(loc="upper right")
    plt.title("Active vs passive calibration with no height")
    plt.xlabel('Height [um]')
    plt.ylabel('Stiffness [pN/nm]');

.. image:: figures/ac_pc_no_height.png

We see that the results are quite dramatically different now.
The stiffness values for passive are much lower, and the passive calibration is as constant as before.
To see why this is the case, let's have a look at the drag coefficient inferred from the active calibration procedure::

    plt.figure()
    drag_x = [calibrations_no_height[h]["ac_x"].measured_drag_coefficient for h in keys]
    drag_y = [calibrations_no_height[h]["ac_y"].measured_drag_coefficient for h in keys]
    plt.plot(heights, drag_x, 'C0x', label="x")
    plt.plot(heights, drag_y, 'C1.', label="y")
    plt.xlabel(r"Height [$\mu$m]")
    plt.ylabel("Drag coefficient [kg/s]")

    # Plot what we expect for the drag coefficient
    sphere_friction_coefficient = 3.0 * np.pi * lk.viscosity_of_water(25) * bead_diameter * 1e-6
    surface_factor = lk.surface_drag_correction(heights, bead_diameter, axial=False)
    plt.plot(heights, surface_factor * sphere_friction_coefficient, color="k", linestyle="--")

.. image:: figures/ac_drag.png

What we can see is that the drag coefficient changes steeply as a function of distance to the surface. Not taking into account the height above the surface results in an incorrectly assuming drag coefficient for passive calibration, resulting in a very different value for the trap stiffness.

Plotting the estimated drag coefficients with the model also gives a clue on why our result in the passive case is slightly off. The height above the surface is not exactly correct!

Try playing a little with the heights and bead diameter, to see how strongly these two affect both the drag coefficient and the trap stiffness::

    %matplotlib widget
    from ipywidgets import interact, FloatSlider

    plt.figure()

    def plot_curve(height_error, bead_diameter_error):
        plt.clf()
        plt.plot(heights, drag_x, 'C0x', label="x")
        plt.plot(heights, drag_y, 'C1.', label="y")
        plt.xlabel(r"Height [$\mu$m]")
        plt.ylabel("Drag coefficient [kg/s]")

        # Plot what we expect for the drag coefficient
        sphere_friction_coefficient = 3.0 * np.pi * lk.viscosity_of_water(25) * (bead_diameter + bead_diameter_error) * 1e-6
        surface_factor = lk.surface_drag_correction(
            heights + height_error + bead_diameter_error / 2,
            bead_diameter + bead_diameter_error,
            axial=False,
        )
        plt.plot(heights, surface_factor * sphere_friction_coefficient, color="k", linestyle="--")


    interact(
        plot_curve,
        height_error=FloatSlider(min=-0.2, max=0.2, step=0.01, value=0),
        bead_diameter_error=FloatSlider(min=-0.2, max=0.2, step=0.01, value=0)
    );

.. image:: figures/widget_surf.png
