.. _apireference:

=============
API Reference
=============

.. currentmodule:: tfdatacompose

Base classes
============
These classes can be implemented to create your data operations.

Map
---
.. autosummary::
    :nosignatures:
    :toctree: generated/

    map.map.Map
    map.numpymap.NumpyMap
    map.pythonfunctionmap.PythonFunctionMap
    flatmap.FlatMap

Filter
------
.. autosummary::
    :nosignatures:
    :toctree: generated/

    pipeline.Pipeline
    filter.filter.Filter
    filter.numpyfilter.NumpyFilter
    filter.pythonfunctionfilter.PythonFunctionFilter


Compose
-------
.. autosummary::
    :nosignatures:
    :toctree: generated/

    pipeline.Pipeline
    datasetoperation.DatasetOperation

Operations
----------
These operations are ready to be used.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    batch.Batch
    skip.Skip
    take.Take

Utilities
---------
These operations can help you debug your pipelines.

.. autosummary::
    :nosignatures:
    :toctree: generated/

    util.print.Print
    util.printshape.PrintShape
