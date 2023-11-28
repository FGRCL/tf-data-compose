.. _apireference:

=============
API Reference
=============

.. currentmodule:: tfdatacompose

Base classes
============
Base classes to create your data operations.

Map
---
.. autosummary::
    :toctree: generated/
    :template: class.rst

    map.map.Map
    map.numpymap.NumpyMap
    map.pythonfunctionmap.PythonFunctionMap
    flatmap.FlatMap

Filter
------
.. autosummary::
    :toctree: generated/
    :template: class.rst

    filter.filter.Filter
    filter.numpyfilter.NumpyFilter
    filter.pythonfunctionfilter.PythonFunctionFilter


Compose
-------
.. autosummary::
    :toctree: generated/
    :template: class.rst

    pipeline.Pipeline
    datasetoperation.DatasetOperation

Operations
==========
Ready to use operations.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    batch.Batch
    skip.Skip
    take.Take

Utilities
=========
Operations to help debug with debugging pipelines.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    util.print.Print
    util.printshape.PrintShape
