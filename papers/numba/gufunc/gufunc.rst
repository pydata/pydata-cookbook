Generalized Ufuncs
------------------

In a previous section, we saw the power of ufuncs for expressing functions on
arrays as functions on scalars combined with broadcasting.  This covers a
large number of situations, but the elements of the output array can only
depend on one element of each input at time.  If multiple elements from a
single input array need to be combined, some other approach is required.

For concreteness, let's suppose we want to compute a weighted average of the
elements in each row of a 2D array, producing a 1D array as output.  A pure
NumPy approach to this problem would look like:

.. code-block:: python

    def np_weighted_avg(x, weights):
    return (x * weights).sum(axis=1)

We assume that ``weights`` is a 1D array and has already been normalized so
that ``weights.sum() == 1``. For a 10000x100 input array (producing a 10000
element output array), the above function takes 3.6 ms on our test system.

If we thinking about how NumPy works in the Python interpreter, we realize
that the above implementation creates a temporary array with the weighted
values of ``x``, only to immediately throw it away after summing along axis 1.
Numba allows us to express the operation without temporaries:

.. code-block:: python

    @numba.jit(nopython=True)
    def nb_weighted_avg_loops(x, weights):
        result = numpy.empty(x.shape[0], x.dtype)
        for i in range(x.shape[0]):

            row_sum = 0.0
            for v, w in zip(x[i], weights):
                row_sum += v*w
                
            result[i] = row_sum
            
        return result

This is much how one would write this function in C or FORTRAN, although we
can use some Python conviences functions like ``zip`` to more naturally loop
through each row and the ``weights`` array at the same time.  This version of
the function runs in 1.5 ms, or an improvement of 2.6x over NumPy.

Now, we might be content to stop here, but this form of the function has some
disadvantages.  First, the explicit looping limits this implementation to only
work on 2D arrays.  If we had a 3D array (say 1000x1000x100), we would have to
write another version of the above function to operate on 3D arrays.  Second,
it would be nice to make this more like a ufunc so we do not have to manage
allocating the output array, and so we can focus on core piece of the
algorithm: the weighted sum itself.

NumPy has a generalization of the ufunc, called a *gufunc*, that Numba can
also create.  A gufunc is distinguished from a ufunc by having a layout
signature that describes the dimensionality of the inputs that should be
passed to the core function.  The signature is a mini-language described more
fully in the NumPy documentation, though we will give some examples below.  A
regular ufuncs with two input arguments implicitly have a signature of
``(),()->()``, indicating that the core function takes two scalar inputs and
produces a scalar output, for example.

Numba creates gufuncs using the ``@guvectorize`` decorator.  Unlike the
``@vectorize`` decorator for making ufuncs, ``@guvectorize`` always requires a
type signature (along with the gufunc signature) for the function inputs.  If
we wish to compile our weighted average gufunc for float32 and float64 inputs,
we would write the following (explained further below):

.. code-block:: python

    @numba.guvectorize(['(float64[:], float64[:], float64[:])', 
                        '(float32[:], float32[:], float32[:])'],
                       '(i),(i)->()')
    def gufunc_weighted_avg(row, weights, result):
        row_sum = 0.0
        for v, w in zip(row, weights):
            row_sum += v*w
        result[0] = row_sum

Our core function takes three 1D arrays as input: the row, the weights, and
the output.  All gufuncs pass in the output array as the last argument.  This
allows NumPy to allocate the full output array up front, and then pass views
onto slices of the output array into the core function.  Following the
behavior of the underlying C implementation of gufuncs, a Numba gufunc which
reads or writes a scalar must access it via the first element of a 1D array
instead.

When we call this gufunc from our application, we do not have to pass the
result argument present in the core function:

.. code-block:: python

    result = gufunc_weighted_avg(x, w)


The layout signature for this gufunc, ``(i),(i)->()``, indicates that two 1D
arrays of identical length should be passed to the core function, and a scalar
is returned.  The looping over additional dimensions is implicitly handled by
NumPy.  A 3D input array and a 2D weight array could be processed by this
gufunc (producing a 2D output), as long as the length of the last dimension of
each input array was identical.

The type signatures for functions in Numba use a special mini-language which
has the following rules:

  1. Scalar types have the NumPy dtype names: ``float32``, ``int8``, ``complex128``, etc.

  2. Array types use a scalar type and colons in brackets to indicate dimensions:

     * ``int32[:]`` = 1D

     * ``int32[:,:]`` = 2D

     * ``int32[:,:,:]`` == 3D

  3. The input arguments to a function are represented by a tuple of types.

For our test case of a 10000x100 input array, the gufunc version of the
weighted average executes in 0.96 ms, or 4x faster than the original NumPy.
That's already a nice improvement, but we can even go even further with
auto-parallelization, as shown in the next section.
