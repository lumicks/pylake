Installation
============

.. _Python: https://www.python.org/
.. _SciPy: http://www.scipy.org/

Pylake can be installed on Windows, Linux or Mac, with the following prerequisites:

* `Python`_ 3.6 or newer (Python 2.x is not supported)
* The `SciPy`_ stack of scientific packages

If you're already familiar with Python and have the above prerequisites, installing Pylake is just a simple case of using `pip`, Python's usual package manager::

    pip install lumicks.pylake

Alternatively, if you're using Anaconda::

    conda install lumicks.pylake -c conda-forge

If you are new to Python/SciPy, more detailed installation instructions are available below.


Anaconda
--------

.. _Anaconda: https://www.anaconda.com/download/
.. _conda-forge: https://conda-forge.org

The easiest way to install Python and SciPy is with `Anaconda`_, a free scientific Python distribution for Windows, Linux and Mac.

.. rubric:: Windows

#. Go to the `Anaconda`_ website and download the Python 3.6 installer.

#. Run it and accept the default options during the installation.

#. Open `Anaconda Prompt` from the `Start` menu. Enter the following command a press enter. This will enable the `conda-forge`::

    conda config --add channels conda-forge

#. Finally, enter the following command to install Pylake::

    conda install lumicks.pylake

That's it, all done. Check out the :doc:`Tutorial </tutorial/index>` for some example code and Jupyter notebooks to get started.


.. rubric:: Linux

#. Go to the `Anaconda`_ website and download the Python 3.6 installer.

#. Open a terminal window and run::

    bash Anaconda3-x.x.x-Linux-x86_64.sh

   Follow the installation steps. You can accept most of the default values, but make sure
   that you type `yes` to add Anaconda to `PATH`::

       Do you wish the installer to prepend the Anaconda3 install location
       to PATH in your /home/<user_name>/.bashrc ? [yes|no]
       [no] >>> yes

   Now, close your terminal window and open a new one for the changes to take effect.

#. Next, enable `conda-forge`::

    conda config --add channels conda-forge

#. Finally, install Pylake with the following command::

    conda install lumicks.pylake

That's it, all done. Check out the :doc:`Tutorial </tutorial/index>` for some example code and Jupyter notebooks to get started.


.. rubric:: macOS

#. Go to the `Anaconda`_ website and download the Python 3.6 installer.

#. Run it and accept the default options during the installation.

#. Open `Terminal` and run the following command to enable `conda-forge`::

    conda config --add channels conda-forge

#. Finally, install Pylake with the following command::

    conda install lumicks.pylake

That's it, all done. Check out the :doc:`Tutorial </tutorial/index>` for some example code and Jupyter notebooks to get started.


Updating
--------

If you already have Pylake installed and you want to update to the latest version, just run::

    conda update lumicks.pylake


.. _ffmpeg_installation:

Optional dependencies
---------------------

.. rubric:: ffmpeg

Exporting to compressed video formats requires an additional dependency named ffmpeg which must be installed separately.
When using conda, ffmpeg can be installed as follows::

    conda install -c conda-forge ffmpeg


Troubleshooting
---------------

If you run into any errors after installation, try updating all conda packages to the latest versions using the following command::

    conda update --all


Frequently asked questions
--------------------------

**Why are the plots in my notebook not interactive?**

To enable interactive plots, you have to invoke the correct `magic commands <https://ipython.readthedocs.io/en/stable/interactive/magics.html>`_
in the notebook. When using Jupyter notebook, the following command will switch the `matplotlib` backend from the inline
one (which renders images) to the interactive backend::

    %matplotlib notebook

You can also choose to install `ipympl`, which can perform better in some cases. You can install it with `pip`::

    pip install ipympl

or `conda`::

    conda install -c conda-forge ipympl

The `ipympl` backend can be activated by invoking the following magic command in a notebook::

    %matplotlib widget

*Note that switching backends typically requires you to restart the Jupyter kernel*. When using JupyterLab, `ipympl` is
the only backend that provides interactive plots with Pylake.


**Conda takes a long time to resolve the environment and then fails. What can I do?**

Several packages depend on each other. Sometimes, finding a suitable collection of packages that is compatible can be
problematic. One way to work around this is to make a separate environment for working with `pylake`. You can create a
new environment named `pylake` by invoking the following in the anaconda prompt::

    conda create -n pylake

The environment can then be activated by calling `activate`::

    conda activate pylake

You can see that it is activated, because `pylake` should now be prefixed to the path in your anaconda prompt. We can
install `pylake` and `jupyter notebook` in this environment by invoking the following commands::

    conda install -c conda-forge lumicks.pylake
    conda install -c conda-forge jupyter notebook

It should be possible to open a jupyter notebook in this environment by calling `jupyter notebook`. Note that if you
are used to starting Jupyter from the anaconda navigator, you will have to set the environment to `pylake` for it to
have access to the pylake package.


**How do I check which version of pylake I have?**

From within `python` or a `notebook` you can invoke::

    import lumicks.pylake as lk
    lk.__version__

Which should return the version number.


**How do I know whether Pylake installed correctly?**

You can run the test suite as follows::

    import lumicks.pylake as lk
    lk.pytest()

If all tests pass (except for the slow ones which are skipped) then your installation of `pylake` is good to go.
