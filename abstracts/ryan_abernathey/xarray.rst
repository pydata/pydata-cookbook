:author: Ryan Abernathey
:email: rpa@ldeo.columbia.edu
:institution: Columbia University, New York, NY, USA
:institution: Lamont Doherty Earth Observatory, Palisades, NY, USA
:corresponding:

:author: Joe Hamman
:email: jhamman1@uw.edu
:institution: Department of Civil & Environmental Engineering,
  University of Washington, Seattle, WA

:author: Stephan Hoyer
:email: shoyer@gmail.com
:institution: Google Research, Mountain View, CA, USA

-------------------------------------------------
xarray: N-D labeled arrays and datasets in Python
-------------------------------------------------

.. class:: abstract

   xarray is an open source project and Python package that provides a toolkit
   and data structures for N-dimensional labeled arrays. The Common Data Model,
   an abstract data model for scientific datasets, is the foundation for
   xarray data stuctures. By combining the array-manipuation capabilites of
   NumPy, the out-of-core computations of dask, and the indexing and grouping
   functionality of Pandas, xarray streamlines and accelerates a wide range of
   data-science workflows, particularly when dealing with gridded datasets
   common in geosciences. Serialization to and from common storage formats
   (e.g. netCDF, HDF, etc.) is also supported. This paper reviews the basic
   xarray api and then dives into some examples from climate science.

.. class:: keywords

   Data Analysis, Python, Pandas, dask, netCDF

Introduction
------------

Many categories of data can be represented as multidimensional (i.e.
N-dimensional) numeric arrays. One-dimensional data are produced by continuous
point measurements, such as timeseries of temperature from a weather station.
Two dimensional data is associated with images and is commonly produced via
remote sensing (aircraft and satellites). Such data can be actual visible
imagery or other fields such as surface temperature, soil moisture, elevation,
etc. When two-dimensional image data is gathered over time, the resulting
dataset becomes three-dimensional. Four dimensions are realized by earth system
models, which simulate how a three-dimensional system evolves in time. Finally,
arbitrary numbers of dimensions result when ensembles of data products are
concatenated, compared, and synthesized in one analysis. (Our examples
are mostly drawn from environmental sciences, where xarray was adopted early on,
but many other fields will recognize parallels, including physics, astronomy,
finance, and biomedical imaging.)

The day-to-day work of scientists and researchers in many fields consists of
organizing, analyzing, and visualizing N-dimensional array data. Within the
python ecosystem, ``numpy`` and ``matplotlib`` have been the backbone of these
workflows for more than a decade. ``xarray`` provides a layer on top of these
tools which greatly simplifies and accelerates workflows, resulting in cleaner,
more-readable code; more intuitive, data-aware syntax; and intelligent
visualization. Furthermore, by integrating with ``dask`` (link to dask
chapter?), ``xarray`` facilitates the analysis of very large datasets without
forcing the user to learn specialized "big data" tools.

A central concept behind ``xarray`` is the notion that *most data is labelled*.
For example, a weather station might gather the variables ``temperature`` and
``humidity`` over the dimension ``time``. These labels are part of the data's
"metadata." It is unfortunately common practice in scientific data analysis to
discard metadata during the data-processing phase. This can lead to bugs and
misinterpretations and creates a barrier to reproducibility. ``xarray`` keeps
the data's labels (and possibly other metadata) together with the raw data
itself for the duration of the workflow, from ingestion through processing to
visualization.

This chapter first introduces basic ``xarray`` usage, including the data model
indexing operators, computation, and grouping operators.We then discuss how to
load common data formats, with a focus on common problems and pitfalls. Finally,
we have an in-depth example which demonstrates advanced usage.

Basic Usage
-----------

The Data Model
^^^^^^^^^^^^^^

Do we just copy the xarray docs here? Is that even allowed by copyright?

Indexing
^^^^^^^^

GroupBy: split-apply-combine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Reading and Writing Data
------------------------

Backends and Engines
^^^^^^^^^^^^^^^^^^^^

netCDF, HDF, RasterIO, OpenDAP
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Writing
^^^^^^^

We need to address compression, chunking, out-of-core!


Advanced Example
----------------

Load a global surface temperature dataset. Remove seasonal cycle. Detrend.
Apply EOF analysis. (Can we use ``apply``?)
