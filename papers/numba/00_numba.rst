:author: Siu Kwan Lam
:email: siu@continuum.io
:institution: Continuum Analytics

:author: Antoine Pitrou
:email: antoine.pitrou@continuum.io
:institution: Continuum Analytics

:author: Stan Seibert
:email: stan.seibert@continuum.io
:institution: Continuum Analytics

------------------------------------------------
Numba: A Compiler for Python and NumPy
------------------------------------------------

.. class:: abstract

   Numba is a just-in-time compiler for Python functions aimed primarily at numerical applications.  Numba performs automatic type inference and generates optimized machine code using the LLVM compiler toolkit.  Numba includes built-in support for NumPy arrays and many standard NumPy functions, but can be extended to support additional data types and custom operations.  The Numba compiler can produce several different kinds of executable objects such as regular functions, NumPy universal functions (ufuncs), and GPU compute kernels.  In addition, Numba can automatically parallelize ufuncs for execution on multicore CPUs and GPUs.

.. class:: keywords

   compiler, hpc, numpy

.. include:: output/numba/intro/intro.rst

.. include:: output/numba/array_vs_list/array_vs_list.rst

.. include:: output/numba/piecewise/piecewise.rst

.. include:: output/numba/looplifting/looplifting.rst

.. include:: output/numba/nogilthreads/nogilthreads.rst
