Piecewise Functions
-------------------

Let's suppose we are creating some kind of scoring function where we need to
compute a piecewise penalty function on every element of an input array.  The
function looks like:

.. math::

    f(x) = \begin{cases}
    x^2 & \text{if $x < 0$}\\
    0 & \text{if $0 \le x \le 1$}\\
    (x-1)^2 & \text{if $x > 1$}
    \end{cases}

A first attempt to efficiently implement this function for a NumPy array
without Numba might look like:

.. code-block:: python

    def np_manual_piecewise(x):
        output = numpy.empty_like(x)
        
        selector = x < 0
        output[selector] = x[selector]**2

        selector = (0 <= x) & (x <= 1)
        output[selector] = 0
        
        selector = x > 1
        output[selector] = (x[selector] - 1)**2
        
        return output

In the above, we compute a boolean array ``selector`` that is used to
selectively apply the right expressions to different parts of the array.

This code pattern is common enough that NumPy offers a standard function, 
``numpy.piecewise``, that allows us to rewrite the above as:

.. code-block:: python

    def np_piecewise(x):
        return numpy.piecewise(x,
                              [x < 0, 
                               (0 <= x) & (x <= 1), 
                               x > 1],
                              [lambda v: v**2,
                               lambda v: 0, 
                               lambda v: (v - 1)**2])

Using ``numpy.piecewise`` makes the code much more compact, but requires
listing the boolean selector arrays separate from the expressions used for
each component.  It is also 30% slower than the previous implementation.

We can also use Numba to implement our piecewise function.  In fact, the first
version can be compiled directly by Numba, just by adding a ``@jit`` 
decorator:

.. code-block:: python

    @numba.jit
    def nb_manual_piecewise(x):
        output = numpy.empty_like(x)
        
        selector = x < 0
        output[selector] = x[selector]**2

        selector = (0 <= x) & (x <= 1)
        output[selector] = 0
        
        selector = x > 1
        output[selector] = (x[selector] - 1)**2
        
        return output

This speeds up the function by a factor of 2, thanks to Numba's automatic
compilation of array expressions.  However, we can improve the speed and
readability of this function by changing it into a NumPy "universal function",
also known as a "ufunc".

A ufunc is a function with scalar inputs and outputs that can be automatically
*broadcast* over one or more input arrays.  The scalar function is computed
for each element in the input arrays, creating an output array.  Most of the
built-in math functions in NumPy are actually ufuncs.

Numba has the ability to create new ufuncs by applying the ``vectorize``
decorator to a scalar function.  For example, we can implement our piecewise
function like this:

.. code-block:: python

    @numba.vectorize
    def nb_piecewise(x):
        if x < 0:
            return x**2
        elif x <= 1:
            return 0
        else:
            return (x - 1)**2

Like ``@jit``, we do not need to specify the data types[#]_ of the input.
This implementation is much more readable, and for our test case of 50,000
input elements, it is 24x faster than the original!

.. [#] Types are needed for ``@vectorize`` when automatically using compilation targets like ``"parallel"`` or ``"cuda"``.


This miraculous-seeming result is a result of rewriting the function so that
fewer temporary arrays need to be allocated.  Memory allocation can be quite
slow, and the first implementation of this piecewise function needed to
allocate 8 temporary arrays.  The ``@vectorize``-based implementation only
allocates one array: the output array.

.. table:: Timings for piecewise function execution on a 50,000 element input. :label:`piecewise-times`

   +----------------------+---------------------------+
   | Function             | Time                      |
   +======================+===========================+
   | np_manual_piecewise  | :math:`334\mu\text{s}`    |
   +----------------------+---------------------------+
   | np_piecewise         | :math:`431\mu\text{s}`    |
   +----------------------+---------------------------+
   | nb_manual_piecewise  | :math:`178\mu\text{s}`    |
   +----------------------+---------------------------+
   | nb_piecewise         | :math:`14\mu\text{s}`     |
   +----------------------+---------------------------+
