Loop Lifting
------------

In a real application, functions may depend on features that Numba has no
optimization strategy.  Such functions will be compiled by Numba using the
*object mode* [#]_, which entirely relies on the *object protocol* in the Python
C-API.  In effect, this is equivalent to unrolling and specializing the
interpreter loop to the bytecode of the function.  The performance advantage
is minimal.

.. [#] https://docs.python.org/3.5/c-api/object.html

Fortunately, the loops in numerical code are usually the hotspots.
They are more likely to contain Python and Numpy features that Numba can
optimize.  Leveraging this simple insight, Numba can optimize functions that
cannot be fully compiled into fast machine code.  We call this *Loop-lifting*,
which extracts ("lifts") loops in functions and transforms them into new
functions for deferred compilation.  When the extracted loops are executed,
the compiler can use the new information available at runtime to further
optimize the loop into efficient machine code.

Suppose we have a simple binning operation we applied to an array.  The binning
function looks like:

.. code-block:: python

    @numba.jit
    def assign_bin(val):
        limit = 32
        while limit < 2**12:
            if val < limit:
                return limit
            limit *= 2
        return limit

    @numba.jit
    def binning(ary):
        bins = numpy.zeros(ary.size, dtype=numpy.int32)
        for i, v in enumerate(ary):
            bins[i] = assign_bin(v)
        return bins


The ``binning`` function returns a new array of the bin assignment for each
value in the input array.  The ``assign_bin`` function assigns values to bins
of power-of-twos between 32 and 4096.  Both functions are marked for compilation
with ``@numba.jit``.

Let's suppose ``binning`` is used in a larger application and we want to insert
some debugging code to investigate a problem.  So, we modify it to the
following:

.. code-block:: python

    @numba.jit
    def binning(ary, debug=False):
        bins = numpy.zeros(ary.size, dtype=numpy.int32)
        for i, v in enumerate(ary):
            bins[i] = assign_bin(v)
        # New debug code
        if debug:
            ctr = collections.defaultdict(int)
            for k in bins:
                ctr[k] += 1
            pprint.pprint(ctr)
        return bins


The newly added if-branch would allow us to inspect the number of items in each
bin.  However, Numba has no strategy in generating machine code for the usage
of ``collections.defaultdict`` and ``pprint.pprint``.  It will fallback to
*object mode*. Fortunately, loop-lifting will "lift" the loop that calls
``assign_bin`` and generate fast code for it. Without loop-lifting, the entire
function will execute in *object mode*.

.. table:: Timings for ``binning`` function execution on a 10,000 element input
           with different optimization settings.
           :label:`looplifting-binning-times`

   +----------------------+---------------------------+
   | Optimization Setting | Time                      |
   +======================+===========================+
   | interpreted          | :math:`6.04\text{ms}`     |
   +----------------------+---------------------------+
   | no looplift          | :math:`5.85\text{ms}`     |
   +----------------------+---------------------------+
   | looplift             | :math:`0.04\text{ms}`     |
   +----------------------+---------------------------+

Table :ref:`looplifting-binning-times` shows the execution timings for the
``binning`` function under different optimization setting for a 10,000 element
input array. With the *interpreted* setting, the function is executed by the
CPython interpreter and no optimization are applied. With the *no looplift*
setting, the entire function is compiled using the *object mode* and the loop
is not lifted. Its performance is similar to the *interpreted* setting.
The *looplift* setting represents the optimal case where the loop is lifted
and compiled into fast machine code.  It runs at 100 times faster than the
other two settings.

Even though *loop-lifting* is convenient for quickly optimizing numerical loops
in arbitrary functions with no code modification, it does not provide immediate
feedback when the loops fail to be optimized.  For *loop-lifting* to work, the
loop must not contain any code that would trigger *object mode*. For example,
the following variation will run slow:

.. code-block:: python

    @numba.jit
    def binning(ary, debug=False):
        bins = numpy.zeros(ary.size, dtype=numpy.int32)
        ctr = collections.defaultdict(int)
        for i, v in enumerate(ary):
            b = assign_bin(v)
            ctr[b] += 1   # uses a defaultdict
            bins[i] = b
        pprint.pprint(ctr)
        return bins

The reference to ``ctr``, which is a ``defaultdict``, in the loop forces the
loop to run in *object mode*.  User can inspect the types for each statement
by calling ``binning.inspect_types()`` to get source code with type annotation
and manually checks for the abscence of Python object inside the loop.  Shown
below is an example output for loop in the previous function:

.. code-block:: python

    for i, v in enumerate(ary):

        # --- LINE 22 ---
        #   $58.5 = global(assign_bin: ...)  :: pyobject
        #   $58.7 = call $58.5(v)  :: pyobject
        #   b = $58.7  :: pyobject

        b = assign_bin(v)

        # --- LINE 23 ---
        #   $58.12 = getitem(...)  :: pyobject
        #   $const58.13 = const(int, 1)  :: pyobject
        #   $58.14 = inplace_binop(...)  :: pyobject
        #   ctr[b] = $58.14  :: pyobject

        ctr[b] += 1   # uses a defaultdict


Each line in the Python source code is preceded with the corresponding internal
representation encoded as comments.  The right-hand-side of "::"
indicates the output type of the operation.  A ":: pyobject" indicates the use
of `PyObject` and the use of *object mode*.

If performance is critical, users are advised to manually extract any loop into
a separate function and decorate the function with ``@numba.jit(nopython=True)``.
The ``nopython`` flag will cause an exception to be raised if the function
fallbacks to *object mode*.


.. comment: maybe add a discussion the numba html annotate feature.

