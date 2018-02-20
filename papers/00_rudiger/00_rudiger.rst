:author: Philipp J. F. Rudiger
:email: philippjfr@continuum.io
:institution: Continuum Analytics, Inc.
:equal-contributor:

:author: Jean-Luc R. Stevens
:email: jstevens@continuum.io 
:institution: Continuum Analytics, Inc.
:institution: University of Edinburgh
:equal-contributor:

:author: James A. Bednar
:email: jbednar@continuum.io
:institution: Continuum Analytics, Inc.
:institution: University of Edinburgh
:corresponding:

:video: https://www.youtube.com/watch?v=0jhUivliNSo

-------------------------------------------------
HoloViews: Visualization
-------------------------------------------------

.. class:: abstract

   HoloViews_ provides a high-level declarative Python API to
   encapsulate multidimensional datasets in a form that can be
   instantly visualized and analyzed.  HoloViews allows you to specify
   even complex animated, multi-figure layouts using just a couple of
   lines of Python code, producing publication-quality plots rendered
   using Matplotlib.  You can just as easily build highly interactive
   web applications with sliders, selections, and streaming data,
   rendered using Bokeh_. You can now fill your Jupyter notebooks with
   informative, interactive visualizations specified clearly and
   succinctly, rather clogging it with pages and pages of
   domain-specific plotting code that is impossible to read or
   maintain.  HoloViews_ (and its companion geographic library
   GeoViews_) makes it simple to designate how your data should
   appear, and then let you index, select, sample, slice, reduce, and
   aggregate it to show just the data you are interested in.  Every
   HoloViews object preserves the original data from which it was
   constructed, allowing you to work as naturally and flexibly with a
   visualization in Python as you can with the original dataset,
   always being able to continue visualizing and analyzing at any
   point without having to replay your steps.  These features all
   derive from the way that HoloViews cleanly separates the plotting
   details from the semantic aspects of your data (*how* your plot
   should appear vs. *what* it shows), making it much simpler to work
   with both your data and your visualizations over time.  In this
   chapter we show how to create HoloViews objects from source data in
   NumPy, Pandas, and Xarray objects, how to manipulate them
   to show different aspects of the data, how to flexibly customize
   how the plots appear, and how to build dynamic, interactive
   visualizations using minimal code for maximal effect.
   
.. _HoloViews: http://holoviews.org
.. _GeoViews: http://geo.holoviews.org
.. _Matplotlib: http://matplotlib.org
.. _Bokeh: http://bokeh.pydata.org

.. class:: keywords

   Visualization, Plotting, Python, Declarative APIs, Bokeh, Matplotlib
