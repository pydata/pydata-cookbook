:author: James Crist
:email: jcrist@continuum.io
:institution: Continuum Analytics

:author: Matthew Rocklin
:email: mrocklin@gmail.com
:institution: Continuum Analytics

----
Dask
----

.. class:: abstract

Dask is a general purpose library for parallel computing that is designed to
complement other PyData libraries.  At its core, Dask is a task scheduling
system, similar to libraries like Luigi or Airflow, but designed for the
computational and interactive workloads found in data science and analytic
problems.

On top of this core, Dask provides parallel and larger-than-memory versions of
NumPy, Pandas, and lists by coordinating operations on many of these objects
with the task schedulers.  For example one logical dask.array is comprised of a
grid of NumPy arrays and one logical dask.dataframe is a sequence of Pandas
dataframes.  Dask includes the parallel algorithms necessary to faithfully
implement a broad and commonly used subset of the functionality of those
libraries.

This chapter will cover both the use of the Dask collections like Dask.array
and Dask.dataframe, and also the use of the core task scheduler, for parallel
problems that do not fit into either of these molds.  This chapter discusses
both scaling up on a single machine, expanding workable data sizes from "fits
in memory" to "fits on disk" as well as the use of Dask on a cluster.
