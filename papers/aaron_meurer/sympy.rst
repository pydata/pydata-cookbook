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
============

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
===========

Gotchas
-------

A primary design goal of SymPy is to act as a first-class Python citizen. This
means that all operations in SymPy use idiomatic Python as much as possible.
Users familiar with the Python language should fell at home in SymPy, although
those coming from other computer algebra systems may be surprised by some of
the design decisions. As the reader of this book is expected to be familiar
with Python, we will point out some of the most relevant gotchas, but not
dwell on them.

The first thing is that all all symbolic variables, called "symbols" in SymPy
parlance, must be defined manually. This is done with the ``symbols``
function.

.. code-block:: python

   >>> y + 1
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
   NameError: name 'y' is not defined
   >>> y = symbols('y')
   >>> y + 1
   y + 1

Secondly, SymPy uses Python's operator overloading, meaning that expressions
can be written more or less as you would any Python code.

.. code-block:: python

   >>> 3*y**2/4 + y - 1
   3*y**2/4 + y - 1

However, one minor gotcha is that Python will evaluate an expression such as
``1/2`` as a float, while in SymPy an exact rational number is often
preferred.

.. code-block:: python

   >>> y + 1/2
   y + 0.5
   >>> y + Rational(1, 2)
   y + 1/2


The third major gotcha to be aware of concerns Python's ``==`` operator. As a
design decision, the ``==`` operator in SymPy always returns a boolean
``True`` or ``False``. Furthermore, this result is strictly based on the
*structural* equality of the two expressions, not the *mathematical* equality.
For example,

.. code-block:: python

   >>> (x + 1)**2 == (x + 1)**2
   True
   >>> (x + 1)**2 == x**2 + 2*x + 1
   False

The two expressions :math:`(x + 1)^2` and :math:`x^2 + 2x + 1` are
mathematically identical, but as SymPy objects, they are different. We can see
they have different types:

.. code-block:: python

   >>> type((x + 1)**2)
   <class 'sympy.core.power.Pow'>
   >>> type(x**2 + 2*x + 1)
   <class 'sympy.core.add.Add'>

To test the mathematical equivalence of expresions, one can subtract them and
call ``simplify``, checking if the result is ``0``:

.. code-block:: python

   >>> simplify((x + 1)**2 - (x**2 + 2*x + 1))
   0

There is also a method ``equals`` which tests the two expressions numerically
at random points.

.. code-block:: python

   >>> ((x + 1)**2).equals(x**2 + 2*x + 1)
   True

Neither method is foolproof. In general, it is mathematically impossible to
prove if two expressions are identically equal or not, so any routine to do
this in SymPy must be fundamentally heuristical in nature.
