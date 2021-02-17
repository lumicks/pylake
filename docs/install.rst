Installation
============

Installation instructions
-------------------------

.. _Python: https://www.python.org/
.. _SciPy: http://www.scipy.org/

Pylake can be installed on Windows, Linux or Mac, with the following prerequisites:

* `Python`_ 3.6 or newer (Python 2.x is not supported)
* The `SciPy`_ stack of scientific packages

If you're already familiar with Python and have the above prerequisites, installing Pylake is just a simple case of using `pip`, Python's usual package manager::

    pip install lumicks.pylake

Alternatively, if you are using Anaconda, please see the more detailed instructions below.


Anaconda
--------

.. _Anaconda: https://www.anaconda.com/download/
.. _conda-forge: https://conda-forge.org

The easiest way to install Python and SciPy is with `Anaconda`_, a free scientific Python distribution for Windows, Linux and Mac.

.. rubric:: Windows

#. Go to the `Anaconda`_ website and download the Python 3.8 (or newer) installer.

#. Run it and accept the default options during the installation.

#. Open `Anaconda Prompt` from the `Start` menu.

#. Create a new environment for Pylake named pylake by invoking the following command::

    conda create -n pylake

#. The environment can then be activated::

    conda activate pylake

#. The line where you enter your next command should now start with the text `(pylake)` instead of `(base)`, to indicate you are in the pylake environment you just created.

#. Pylake can be found on `conda-forge`_. Add `conda-forge`_ to the list of sources as follows::

    conda config --add channels conda-forge

#. We can install Pylake in this environment by invoking the following command::

    conda install lumicks.pylake

#. To make sure Jupyter notebook can find the ipykernel, we invoke::

    python -m ipykernel install --user

#. It should be possible to open a jupyter notebook in this environment by calling::

    jupyter notebook

Note that if you are used to starting Jupyter notebook from the Anaconda Navigator, you will have to set the environment to `pylake` for it to have access to the environment where you just installed Pylake.

You can do this in the drop down menu that normally defaults to the environment `base`.

If you are used to starting Jupyter from the Anaconda prompt, then remember to activate the correct environment (pylake) prior to starting the notebook.
You can activate this environment by calling::

    conda activate pylake

That is all. Check out the :doc:`Tutorial </tutorial/index>` for some example code and Jupyter notebooks to get started.


.. rubric:: Linux

#. Go to the `Anaconda`_ website and download the Python 3.8 (or newer) installer.

#. Open a terminal window and run::

    bash Anaconda3-x.x.x-Linux-x86_64.sh

#. Follow the installation steps. Make sure that you type `yes` when prompted with the question whether to run conda init::

    Do you wish the installer to initialize Anaconda 3 by running conda init? [yes|no]
    [no] >>> yes

#. Now, close your terminal window and open a new one for the changes to take effect.

#. Using the terminal, we will create a new environment for Pylake named pylake by invoking the following command::

    conda create -n pylake

#. The environment can then be activated::

    conda activate pylake

#. Pylake can be found on `conda-forge`_. Add `conda-forge`_ to the list of sources as follows::

    conda config --add channels conda-forge

#. We can install Pylake in this environment by invoking the following command::

    conda install lumicks.pylake

#. You can open a Jupyter notebook in this environment by calling `jupyter notebook` from the terminal.

#. You can also now start the Anaconda Navigator by calling `anaconda-navigator` from the terminal.

Note that if you are used to starting Jupyter notebook from the Anaconda Navigator, you will have to set the environment to `pylake` for it to have access to the environment where you just installed Pylake.
You can do this in the drop down menu that normally defaults to the environment `base`.
If you are used to starting Jupyter notebook from the terminal, then remember to activate the correct environment (pylake) prior to starting the notebook.

That is all. Check out the :doc:`Tutorial </tutorial/index>` for some example code and Jupyter notebooks to get started.


.. rubric:: macOS

#. Go to the `Anaconda`_ website and download the Python 3.8 (or newer) installer.

#. Run it and accept the default options during the installation.

#. Open `Terminal`. First, we will create a new environment for Pylake named pylake by invoking the following command::

    conda create -n pylake

#. The environment can then be activated by invoking the following::

    conda activate pylake

#. Pylake can be found on `conda-forge`_. We can add `conda-forge`_ to the list of sources as follows::

    conda config --add channels conda-forge

#. Install Pylake in this environment by invoking the following command::

    conda install lumicks.pylake

#. You can open a jupyter notebook in this environment by calling::

    jupyter notebook

Note that if you are used to starting Jupyter notebook from the Anaconda Navigator, you will have to set the environment to `pylake` for it to have access to the environment where you just installed pylake.
You can do this in the drop down menu that normally defaults to the environment `base`.
If you are used to starting Jupyter from `Terminal`, then remember to activate the correct environment (pylake) prior to starting the notebook.

That is all. Check out the :doc:`Tutorial </tutorial/index>` for some example code and Jupyter notebooks to get started.


Updating
--------

If you already have Pylake installed and you want to update to the latest version, just run::

    conda update lumicks.pylake

Note that this updates the package to the latest version that is compatible with your environment.
It will also attempt to update any dependencies that require an update in order to be compatible with the updated version of pylake.
You can check which version of pylake you have after this procedure by checking the pylake version
from the command prompt (windows) or terminal (macOS/linux)::

    conda list pylake

If for some reason conda fails to update pylake to the latest version, it is usually easier to just remove the pylake environment and reinstall from scratch.
To do this, open a new Anaconda prompt and type::

    conda env remove -n pylake

After which you can re-install pylake using the regular installation instructions above.


.. _ffmpeg_installation:

Optional dependencies
---------------------

.. rubric:: ffmpeg

Exporting to compressed video formats requires an additional dependency named ffmpeg which must be installed separately.
When using conda, ffmpeg can be installed as follows::

    conda install -c conda-forge ffmpeg


Conda environments
------------------
.. _PyCharm: https://www.jetbrains.com/pycharm/download/#section=windows
.. _PyCharm documentation: https://www.jetbrains.com/help/pycharm/conda-support-creating-conda-virtual-environment.html
.. _Jupyter: https://jupyter.org/
.. _uninstall instructions: https://docs.anaconda.com/anaconda/install/uninstall/
.. _VS Code: https://code.visualstudio.com/download
.. _VS Code Environment Instructions: https://code.visualstudio.com/docs/python/environments#_conda-environments
.. _numpy discussion: https://github.com/numpy/numpy/issues/15183#issuecomment-603575874

If you have installed Pylake according to the Installation Instructions for Anaconda, then you should now have a separate environment for your Pylake work.
You may be wondering why we needed to create a new environment for Pylake, and what an environment is.

When using Python, you will quickly find that several packages that you can install depend on each other.
For example, when installing Pylake, you could see that the installation of Pylake required numerous other packages to be installed as well.
Most Python packages are continuously updated by their authors. Sometimes, these authors decide that certain existing functionalities need to be changed.
This means that not all of them will be completely backwards compatible.
Therefore, it can be challenging to find a set of packages and package versions that all work together.

One pragmatic solution to this is to maintain separate Python environments for different projects.
This means that you create independent "copies" of Python and its installed packages, so that the different projects you are working on do not interfere with each other.
Anaconda is one solution to this problem. With Anaconda, you can have multiple installations of Python (with all their installed modules) installed on your computer.
These installations are referred to as environments.

**Why do we install Pylake in a separate environment by default?**

Conda fetches the packages it uses from a channel, these are locations where conda and the Anaconda Navigator search for packages.
The default one is called Anaconda, but Pylake is available on a channel named `conda-forge`_.
Conda forge and Anaconda both have different versions of different packages.
Some of these are not compatible with each other.
This is why it is wise to install Pylake into its own environment, and only source packages from the channel `conda-forge`_ in that environment.
This helps prevent difficulties when trying to come up with a plan to install a package you request.

**How do I set up my other tools to use the correct environment?**

For most programs, it is just a matter of pointing them to the correct environment.
If you prefer using the Anaconda Navigator, you can activate the environment by selecting it from the drop down menu `Applications on`.
By default, the selected environment is `base`.

*PyCharm*

For small data analysis scripts, `Jupyter`_ notebooks can be quite helpful.
For larger projects you may want to switch to an integrated development environment (IDE).
Our recommended tool for working on larger Python projects is `PyCharm`_.
You can install PyCharm by following the default installation instructions.
Next, we have to set up PyCharm so that it finds the correct Conda environment.
For information on how to do this, please refer to the `PyCharm documentation`_.

*Spyder*

If you have installed `Spyder` with the `pylake` environment active, you should also have a start menu entry that reads `Spyder (pylake)`.
Note how Conda typically installs shortcuts indicating the relevant environment between brackets.

*VS Code*

#. Download `VS Code`_ and install it following the default installation instructions.

#. Start it when the installer finishes.

#. Go to the extensions tab (CTRL + SHIFT + X)

#. Enter Python in the search field.

#. Click on the Python plugin (by Microsoft) and install it.

#. Restart VS Code.

#. Open the Command Palette (CTRL + SHIFT + P) and type "Python: Select Interpreter". Here you should choose the pylake environment.

#. Close VS Code.

VS Code should now appear in your Anaconda Navigator list.
Make sure that you selected the pylake environment when starting VS Code from the Anaconda Navigator.
It should now be possible to use pylake in VS Code.

Under Windows, you will need to start VS Code from the Anaconda Navigator for it to use the correct environment.
For more information on how to get it to run without Anaconda Navigator on Windows see the `VS Code Environment Instructions`_ and this `numpy discussion`_.

**Can I use pip with Anaconda?**

`Pip` is a different package manager.
While `conda` does allow you to install `pip` inside a conda environment, there is no guarantee that `pip` packages will be compatible with `conda-forge` packages.
It is therefore wise to choose one package manager as your go-to package manager and only switch when a package you need can only be found on the other package manager.

While using `pip` within conda is perfectly possible, note that if you do decide to go down this route, you should install all packages in that environment via `pip` and none via `conda`.
If you decide to use this configuration then you have to make sure that you install it using the version of `pip` inside your conda environment.
You may experience incompatibility issues if you use a system-wide install of `pip` in conjunction with an active `conda` environment.

On Windows, the easiest way to find out which `pip` you are using is to invoke `where pip` on the anaconda prompt that you are using.
The `pip` executable that will be called when you invoke it from the command prompt will be at the top and should be located in your conda environment.
You can verify this by checking whether the path contains your currently active environment in it.

If you see that `pip` is either not on your path or it is being fetched from a different location, verify whether you have activated the correct conda environment.
You can activate an environment by invoking `conda activate <environment name>`, where `<environment name>` should be replaced with the environment you want to activate.
If you have already activated the correct environment, but you still do not see `pip` being fetched from it then you can install it into this environment by invoking `conda install pip`.


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
problematic. One way to work around this is to make a new environment for working with Pylake. See the installation
instructions for more information.


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


**I tried the installation instructions but conda still won't install pylake**

If creating a new environment does not work then it may be best to uninstall and reinstall conda.
*Note that this means you will lose all the environments you have created!*
Please follow these `uninstall instructions`_ to uninstall conda.
After uninstalling, you should be able to reinstall using the regular installation instructions.


**Conda-forge is very slow in China, what can I do?**

Conda-forge can be slow when accessed from China.
This can be resolved by using a Chinese mirror to install Pylake.
Since there is no mirror for `conda-forge`, Pylake then has to be installed using pip, as outlined below.

If you normally manage your environments with `pip`, you can just invoke::

    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple lumicks.pylake

If you use Anaconda, then it is best to create a new environment for this installation. You can do this as follows::

    conda create -n pylake_pip

Activate the environment as follows::

    conda activate pylake_pip

Install pip in the activated environment by invoking::

    conda install pip

Then install Pylake as follows::

    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple lumicks.pylake

On Windows, you should also make sure that Jupyter notebook can connect to the ipython kernel by invoking::

    conda install pywin32
    python -m ipykernel install --user

Important to note is that packages on `conda` and `pip` are typically *not* compatible. Therefore, whenever you use this environment, *only* use pip, and do not install additional dependencies via `conda install`, since this can break your environment.
