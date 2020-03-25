Protein FD Fitting
==================

.. only:: html

    :nbexport:`Download this page as a Jupyter notebook <self>`

We first import our required libraries::

    from lumicks import pylake
    import npz
    import numpy as np
    import scipy as sp
    import matplotlib.pyplot as plt
    from ipywidgets import interact, widgets
    
    figx = 9
    figy = 6

First we load our data using pylake.
------------------------------------

We use some of pylake’s built-in downsampling routines to downsample the
data before use (since we do not need the 78.125 kHz data for this
particular analysis)::

    data_filenames = ['AdK/11_07_adk1_2.npz', 
                      'AdK/11_07_adk2_2.npz', 
                      'AdK/11_07_adk5_1.npz', 
                      'AdK/11_07_adk5_2.npz', 
                      'AdK/11_07_adk5_3.npz', 
                      'AdK/11_07_adk5_4.npz'
                     ]
    
    # Load and downsample the data
    downsampling_factor = int(78125/50)
    data_files = [npz.File(data_filenames[idx]) for idx in np.arange(len(data_filenames))]
    data = [{"piezo_distance": data.piezo_distance.downsampled_by(downsampling_factor, reduce=np.median), 
             "piezo_force": data.piezo_force.downsampled_by(downsampling_factor, reduce=np.median)}
            for data in data_files]
    
    plt.figure(figsize=(figx, figy))
    for d in data:
        plt.plot(d["piezo_distance"].data, d["piezo_force"].data, '.')
        plt.grid(True)
        
    plt.xlim([.2, .45])
    plt.xlabel('Distance [$\\mu$m]')
    plt.ylabel('Force [pN]');
    plt.show();


.. image:: output_2_1.png


Determine where the protein unfolds
-----------------------------------

Usually visual inspection remains the most reliable method for
determining where the breaks in the pulling curve are::

    plt.figure(figsize=(figx, figy))
    
    def plot_data(data_idx):
        plt.clf()
        plt.subplot(2, 1, 1)
        data[data_idx]["piezo_distance"].plot()
        plt.subplot(2, 1, 2)
        plt.plot(data[data_idx]["piezo_distance"].data, data[data_idx]["piezo_force"].data, '.')
        plt.xlabel('Distance [$\\mu$m]')
        plt.ylabel('Force [pN]')
        
    interact(plot_data, data_idx=widgets.IntSlider(min=0, max=len(data) - 1, step=1, value=0))


Setting up the global fit.
--------------------------

We make use of two models in this notebook.

-  For the region where the protein has not yet been unfolded, we model
   the Force Extension Curve using a single WLC model.

The Odijk model is given by:

.. math:: d = Lc \left(1 - \frac{1}{2} \sqrt{\frac{k_B T}{F L_p}} + \frac{F}{S_t} \right)

Rather than fitting this model directly with `F` as the independent 
variable, we fit an inverted version of this model that has `d` as 
the independent variable. For a single component WLC model, pylake 
provides an analytically inverted WLC model named ``invWLC``.
This is a fast alternative to using the ``invert`` function.

-  Once the protein unfolds, we need more complexity. Here we make use
   of two extensible WLC models in series to fit this data. In this
   setup the distances are additive and the forces equal.

.. math:: d_{total} = d_{DNA} + d_{protein}

.. math:: F_{total} = F_{DNA} = F_{protein}

In addition to these issues, two of our datasets need an offset, so we
incorporate this into our model. Doing a global fit results in improved
precision and accuracy of parameter estimates as more data is used to
constrain parameters that the datasets have in common. For these
parameters a single, global value is found that holds for all data sets::

    # Construct a model for the DNA. We will use the inverted Odijk model.
    dna_model = pylake.inverted_odijk("DNA").subtract_independent_offset("d_offset") + pylake.offset("f")
    
    # Construct a model for the entire construct.
    construct_model = (pylake.odijk("DNA") + pylake.odijk("protein") + pylake.offset("d")).invert() + pylake.offset("f")
    
    # Set up the fit object, which contains both models
    F = pylake.FitObject(dna_model, construct_model);

First load the data corresponding to the folded state.
------------------------------------------------------
We write a little helper function that helps us load the data. First we 
load the data corresponding to the folded state::

    # Small helper function to load data
    def load_data(model, d, name, time_range, **kwargs):
        start_time = f"{time_range[0]}s"
        end_time = f"{time_range[1]}s"
        force = d["piezo_force"][start_time:end_time].data
        distance = d["piezo_distance"][start_time:end_time].data
        return model.load_data(distance[force < 30], force[force < 30], name=name, **kwargs)
    
    # Folded data
    folded_handles = [
        load_data(dna_model, data[0], "AdK 1", [0, 53], d_offset="d0_offset", f_offset="f0_offset"),
        load_data(dna_model, data[1], "AdK 2", [0, 73], d_offset="d1_offset", f_offset="f1_offset"),
        load_data(dna_model, data[2], "AdK 3", [0, 90], d_offset="d2_offset", f_offset=0),
        load_data(dna_model, data[3], "AdK 4", [0, 88], d_offset="d3_offset", f_offset=0),
        load_data(dna_model, data[4], "AdK 5", [0, 98], d_offset="d4_offset", f_offset=0),
        load_data(dna_model, data[5], "AdK 6", [0, 25], d_offset="d5_offset", f_offset=0)
    ];

Fit the DNA data
----------------

We assign some bounds to the model parameters, to make sure they stay
within reasonable ranges. We want the persistence length of the DNA to 
stay between 29 and 80 for the linker. In addition, we make sure that our
offsets do not go below zero. After setting these bounds, we fit the 
DNA part of our model::


    F.parameters["d0_offset"].lb = 0
    F.parameters["d0_offset"].ub = .4
    F.parameters["d0_offset"].value=.1
    F.parameters["d1_offset"].lb = 0
    F.parameters["d0_offset"].ub = .4
    F.parameters["d1_offset"].value=.1
    F.parameters["f0_offset"].lb = 0
    F.parameters["f1_offset"].lb = 0
    F.parameters["f0_offset"].ub = 2
    F.parameters["f1_offset"].ub = 2
    F.parameters["DNA_Lp"].lb = 35
    F.parameters["DNA_Lp"].ub = 80
    F.parameters["DNA_Lc"].value = .360
    F.parameters["DNA_St"].value = 300
    
    F.fit()
    plt.figure(figsize=(figx, figy))
    plt.xlabel('Distance [$\\mu m$]')
    plt.ylabel('Force [pN]')
    F.plot();


.. image:: output_10_1.png


Add unfolded protein data
-------------------------

Now that we’ve fitted the DNA, we can add data for the model of the
entire construct. This fit takes a bit longer, since it’s a much more
complicated model.

Rather than one analytically inverted model, this model is actually two
models added, which are then inverted.

Now that we actually have added some data for the protein model, its
parameters also become part of the fitting problem. Here we assign
bounds to the protein persistence length, to make sure it stays within a
reasonable range. 

For the protein, we want the persistence length to stay between 1 and 3::

    # Unfolded data
    unfolded_handles = [
        load_data(construct_model, data[0], "AdK 1", [90, 145],  protein_Lc="Lc_unfolded_1", 
                  d_offset="d0_offset", f_offset="f0_offset"),
        load_data(construct_model, data[1], "AdK 2", [120, 145], protein_Lc="Lc_unfolded_2", 
                  d_offset="d1_offset", f_offset="f1_offset"),
        load_data(construct_model, data[2], "AdK 3", [103, 173], protein_Lc="Lc_unfolded_3", 
                  d_offset="d2_offset", f_offset=0),
        load_data(construct_model, data[3], "AdK 4", [93, 184],  protein_Lc="Lc_unfolded_4", 
                  d_offset="d3_offset", f_offset=0),
        load_data(construct_model, data[4], "AdK 5", [101, 171], protein_Lc="Lc_unfolded_5", 
                  d_offset="d4_offset", f_offset=0),
        load_data(construct_model, data[5], "AdK 6", [50, 120],  protein_Lc="Lc_unfolded_6", 
                  d_offset="d5_offset", f_offset=0),
    ]
    
    F.parameters["protein_Lp"].value = 2
    F.parameters["protein_Lp"].lb = 1
    F.parameters["protein_Lp"].ub = 3
        
    plt.figure(figsize=(figx, figy))
    plt.tight_layout(pad=1.08)
    F.fit(max_nfev=126, verbose=True)
    F.plot()
    plt.xlabel('Distance [$\\mu$m]')
    plt.ylabel('Force [pN]');


.. parsed-literal::

    `xtol` termination condition is satisfied.
    Function evaluations 6, initial cost 4.6840e+05, final cost 8.6622e+02, first-order optimality 6.14e+04.
    

.. image:: output_12_2.png

Next, we plot our results::

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(2*figx, 2*figy))
    for i, (d, folded, unfolded) in enumerate(zip(data, folded_handles, unfolded_handles)):
        plt.subplot(2, 3, i + 1)
        distance = d["piezo_distance"].data
        plt.plot(distance, d["piezo_force"].data, 'r.', markersize=4*.8)
        dna_model.plot(F.parameters[folded], distance, fmt='k--')
        construct_model.plot(F.parameters[unfolded], distance, fmt='k--')
        
        plt.grid(True)
        plt.ylabel('Force [pN]')
        plt.xlabel('Distance [$\mu$m]')
        
    plt.xlim([.23, .375])    
    plt.ylim([0, 30])
    plt.tight_layout(pad=1.08)
    plt.savefig('fits_alltogether.eps');
    plt.savefig('fits_alltogether.png', format="png");
    plt.show();


.. image:: output_13_1.png

We make a box plot of the contour length `Lc` of the protein::


    Lcs = [F.parameters[f"Lc_unfolded_{i}"].value*1000 for i in range(1,6)]
    
    plt.figure()
    plt.boxplot(Lcs, labels=' ')
    plt.title('Change in contour length')
    plt.ylabel('$\\Delta  L_c  [nm]$');
    plt.savefig('box.eps');


.. image:: output_14_1.png

