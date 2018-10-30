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


Troubleshooting
---------------

If you run into any errors after installation, try updating all conda packages to the latest versions using the following command::

    conda update --all
