Arrays vs. Lists
----------------

Lists are a very common data structure in Python, providing a way to represent
an ordered sequence of values.  They are incredibly convenient, allowing
elements to be read and replaced quickly.  Elements can be inserted into and
appended to lists as well, though much more slowly.  In fact, the convience of
lists can hide some serious performance drawbacks for numerical code.

The first problem is that lists are more general than is needed for most
numerical algorithms. Lists contain sequences of Python objects, which are
larger in memory than bare numerical values like ``float64`` (8 bytes) or
``int32`` (4 bytes).  In addition, the Python objects corresponding to
elements of a list are not guaranteed to be next to each other in memory, so
loop that process the elements in order may result in non-consecutive memory
accesses, which limit memory bandwidth and frequently result in slower code.

Numba has support for Python lists in nopython mode, with some limitations
that improve performance.  First, all elements of a list in nopython mode must
have the same basic numerical type: integer, floating point, or complex. Numba
will apply its standard casting rules for math expressions to determine this
common type.  Roughly, types of different sizes will be cast to the largest
size, integers and floats cast up to floats, and floats and
complex cast up to complex.  For example:

.. code-block:: python

    @numba.jit(nopython=True)
    def make_list():
        return [1, 2, 3.0]

    print(make_list())

Prints a list of three floats:

.. code-block: python

    [1.0, 2.0, 3.0]

There are some drawbacks to lists in Numba, however.  Because nopython mode
requires all arguments to be translated into machine-native forms (hence the
"nopython" name), any Python list passed into a Numba-compiled function as an
argument must go through an "unwrapping" step which can be quite time
consuming for large lists.  This trivial function:

.. code-block:: python

    @numba.jit(nopython=True)
    def return_first(a_list):
        return a_list[0]

is more than 100x slower in Numba than in pure Python when ``a_list`` is a
1000 element list.

As an alternative to lists, NumPy includes a typed multidimensional array
object (which we'll call a "NumPy array" for short). NumPy arrays can be used
in much the same way as lists in numerical code, but have several major
benefits:

  * They store their data in a native, packed format that is much more space efficient, and memory bandwidth efficient.
  * Numba can "unwrap" a NumPy array very quickly.
  * Multi-dimensional arrays work in Numba, but nested lists are not supported.

In comparison to above, the ``return_first`` function is only 1.9x slower in
Numba than in Python.  Numba continues to be slower in this case because the
function body itself is so simple, the fixed overhead of function dispatch is
dominating the runtime.

Let's consider a situation where we might be tempted to use a list, and see
how a NumPy array might be better.  A common operation when dealing with large
amounts of data is *filtering*.  Suppose we have a million values from a
faulty sensor, where incorrect readings appear with values less than -1000.
Before we do any further statistics on this data, we need to remove the
bad values.

Following the advice given above, we would rightly decide to store the data in
a NumPy array.  Passing the data in as a Python list would be tremendously
slow due to the unwrapping overhead.  However, inside the function we don't
know ahead of time how many values we need to emit, so we might decide to
create an empty list and append to it:

.. code-block:: python

    @numba.jit(nopython=True)
    def filter_bad_try1(values, lower, upper):
        good = []
        for v in values:
            if lower < v < upper:
                good.append(v)
        return numpy.array(good)

For consistency with the rest of our application, we cast the list back into a
NumPy array at the end of the function.  On our development computer, this
function takes 123 ms for ten million values when 50% of them are bad.

We can do better than this, though.  Appending to an array is slow because it
requires expanding the array storage several times to accomodate new values.
It would be much quicker to allocate a temporary NumPy array of the maximum
size required, and write elements into it.  Then we can slice the array to its
final length at the end of the function and return it:

.. code-block:: python

    @numba.jit(nopython=True)
    def filter_bad_try2(values, lower, upper):
        good = numpy.empty_like(values)
        next_index = 0
        for v in values:
            if lower < v < upper:
                good[next_index] = v
                next_index += 1
        return good[:next_index]

This version of the function takes 60 ms, which is a significant improvement.
However, it does have a small drawback.  When Numba returns a slice of a NumPy
array, it returns a view which references the block of memory associated with
the original array.  Normally that's fine, but in the previous example, this
results in wasted memory as the end of the ``good`` array (everything from
``next_index`` to the end) is unused, but not freed until the returned slice
is also freed.  We can fix this by returning a copy of the slice, which will
have no wasted space, and allowing the ``good`` array to go out of scope and
be freed:

.. code-block:: python

    @numba.jit(nopython=True)
    def filter_bad_try3(values, lower, upper):
        good = numpy.empty_like(values)
        next_index = 0
        for v in values:
            if lower < v < upper:
                good[next_index] = v
                next_index += 1
        return good[:next_index].copy()

This does slow down the function slightly to 74 ms, so the `copy()` could be
omitted if speed is more important than memory consumption.

Experienced NumPy users would rightly point that the above algorithm could be
much more compactly expressed using boolean indexing of a NumPy array:

.. code-block:: python

    def filter_bad_try4(values, lower, upper):
        return values[(lower < values) & (values < upper)]

Uncompiled, this function needs to make several temporary arrays to evaluate
the boolean index array, so it requires 96 ms to run on our test dataset. This
is better than the version 1 of the filter function, but not as good as the
second.  Numba can optimize the generated machine code for some array
expressions as well, so this final version of the function:

.. code-block:: python

    @numba.jit(nopython=True)
    def filter_bad_try5(values, lower, upper):
        return values[(lower < values) & (values < upper)]

filters the test set in 60 ms, equivalent to version 3, but with far less
code.
