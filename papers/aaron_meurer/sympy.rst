:author: Aaron Meurer
:email: asmeurer@gmail.com
:institution: University of South Carolina
:corresponding:

-----
SymPy
-----

.. class:: abstract

   SymPy is a symbolic mathematics library written in pure Python. It is a
   full-featured computer algebra system, including functionality for symbolic
   calculus, equation solving, symbolic matrices, code generation, and much
   more, as well as several domain-specific modules including statistics,
   quantum mechanics, and classical mechanics. In this chapter, we give an
   introduction to symbolic computing and the SymPy library, and show through
   an example how it can be used as part of a larger scientific workflow.

.. class:: keywords

   symbolic mathematics, sympy

Introduction
------------

SymPy is a computer algebra system (CAS) library for Python. This means that
it manipulates mathematical expressions in a symbolic way. To understand what
this means, consider this simple example:

.. code-block:: python

   >>> import numpy as np
   >>> import sympy as sym
   >>> np.sqrt(8)
   2.8284271247461903
   >>> sympy.sqrt(8)
   2*sqrt(2)

Unlike NumPy and similar libraries, which are primarily numerical in nature
(the input and output of a function is a number), SymPy functions produce
exact, symbolic representations. We can start to gleam the power of this from
the above example as well. The square root of 8 is simplified as
:math:`2\sqrt{2}`, a fact that is not easily seen from the numeric form.

The power of SymPy comes in its ability to represent arbitrary mathematical
expressions in a purely symbolic way, and to perform mathematical operations
on those expressions. Consider a more complicated example:

.. code-block:: python

   >>> from sympy import *
   >>> x = symbols('x')
   >>> integrate(sin(x)*exp(x), x)
   exp(x)*sin(x)/2 - exp(x)*cos(x)/2

SymPy has computed :math:`\int \sin(x)e^x\,dx` as :math:`e^{x} \sin{\left (x
\right )} / 2 - e^{x} \cos{\left (x \right )} / 2`. SymPy contains functions
to compute symbolic integrals, derivatives, limits, equation solving,
matrices, and code generation, to only name a few (a full list of SymPy
features can be found at http://www.sympy.org/en/features.html).

Using SymPy
-----------
