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
