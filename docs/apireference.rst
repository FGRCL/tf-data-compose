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
These operations will let you transform your data.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    map.map.Map
    map.numpymap.NumpyMap
    map.pythonfunctionmap.PythonFunctionMap
    flatmap.FlatMap

Lambda Map
----------
Shorthand version of the mapping operations.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    lambdamap.lambdamap.LambdaMap
    lambdamap.numpylambdamap.NumpyLambdaMap
    lambdamap.pythonfunctionlambdamap.PythonFunctionLambdaMap


Filter
------
These operations will let you filter your data.

.. autosummary::
    :toctree: generated/
    :template: class.rst

    filter.filter.Filter
    filter.numpyfilter.NumpyFilter
    filter.pythonfunctionfilter.PythonFunctionFilter


Compose
-------
These operations will help you compose your transoformations together to create your pipelines.

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
