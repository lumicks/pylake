.. title:: Pylake: Data analysis for single-molecule optical tweezer data

.. container:: lkheader

    .. image:: logo_notext.png
        :class: lklogo

    Pylake is a Python package for analyzing single-molecule optical tweezer data.
    The :doc:`tutorial/index` section is a good place to start.
    It gives a good overview of the most important features with lots of code examples.
    The source code repository is `located on Github <https://github.com/lumicks/pylake>`_
    where you can also post any questions, comments or issues that you might have.


.. to build html, go in anaconda prompt and type python -m sphinx -b html docs build

.. toctree::
    :hidden:
    :titlesonly:

    changelog


.. toctree::
    :caption: User guide
    :maxdepth: 2

    install
    tutorial/index
    examples/index


.. toctree::
    :caption: Reference docs
    :maxdepth: 2

    api

.. toctree::
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