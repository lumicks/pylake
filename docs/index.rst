.. title:: Pylake: Data analysis for single-molecule optical tweezer data

.. container:: lkheader

    .. image:: logo_notext.png
        :class: lklogo

    Pylake is a Python package for analyzing single-molecule optical tweezer data.
    The :doc:`tutorial/index` section is a good place to start.
    It gives a good overview of the most important features with lots of code examples.
    The source code repository is `located on Github <https://github.com/lumicks/pylake>`_
    where you can also post any questions, comments or issues that you might have.

.. grid:: 1
    :gutter: 3

    .. grid-item-card:: Get Started
        :link: installation-index
        :link-type: ref

        Check out the Installation Guide for easy setup instructions and FAQs.

.. grid:: 3
    :gutter: 3

    .. grid-item-card:: What's New
        :link: whatsnew-index
        :link-type: ref

        Keep up to date with the latest and greatest features introduced in new versions!

    .. grid-item-card:: Tutorials
        :link: tutorials-index
        :link-type: ref

        We have full tutorials demonstrating the major functionality included in Pylake.

    .. grid-item-card:: Fundamentals
        :link: fundamentals-index
        :link-type: ref

        Take a deep dive into some single-molecule topics -- from the basics to advanced theory.

    .. grid-item-card:: Examples
        :link: examples-index
        :link-type: ref

        Full example scripts for analyzing data from different applications.

    .. grid-item-card:: API References
        :link: api-index
        :link-type: ref

        A detailed reference list of all the classes and functions available in Pylake.

    .. grid-item-card:: Literature
        :link: literature-index
        :link-type: ref

        Citations for all of the primary literature used in Pylake analysis implementation.

.. to build html, go in anaconda prompt and type python -m sphinx -b html docs build

.. toctree::
    :hidden:
    :titlesonly:

    changelog
    whatsnew/index

.. toctree::
    :hidden:
    :caption: Getting Started

    install
    examples/index
    api

.. toctree::
    :hidden:
    :caption: Tutorials
    :maxdepth: 2

    tutorial/file
    tutorial/fdcurves
    tutorial/scans
    tutorial/kymographs
    tutorial/imagestack
    tutorial/fdfitting
    tutorial/nbwidgets
    tutorial/kymotracking
    tutorial/force_calibration
    tutorial/population_dynamics
    tutorial/piezotracking

.. toctree::
    :hidden:
    :caption: Fundamentals
    :maxdepth: 1

    theory/diffusion/index
    theory/force_calibration/index

.. toctree::
    :hidden:
    :caption: Literature
    :maxdepth: 2

    zrefs.rst

.. only:: html

    * :ref:`genindex`


.. raw:: html

    <style type="text/css" >
        @media screen and (min-width: 990px) {
            .rst-content p { clear: both; }

            .lklogo {
                margin: 20px 10px 16px 10px;
                float: left;
                width: 105px;
            }

            .lkheader p {
                margin-left: 140px;
                padding-top: 2%;
                padding-bottom: 2%;
                clear: none;
            }
        }
    </style>
