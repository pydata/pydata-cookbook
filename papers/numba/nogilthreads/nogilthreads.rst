Release the GIL for Parallel Execution
--------------------------------------

When a loop is data indenpdent, each iteration has no dependency on each other
and the iteration can be completed in arbitrary order.
An independent can be trivally parallelized.

Let's consider the following function:

.. code-block:: python

    @numba.jit
    def mandel(x, y, max_iters=20):
        i = 0
        c = complex(x, y)
        z = 0.0j
        for i in range(max_iters):
            z = z ** 2 + c
            if (z.real ** 2 + z.imag ** 2) >= 4:
                return i
        return max_iters

The ``mandel`` function determines if a given real and imaginary parts
(as ``x`` and ``y`` arguments) of a complex number is a candidate for membership
in the Mandelbrot set given a fix number of iterations (``max_iters``).
The function is *pure* given that its output is solely determined by the inputs,
and there are no side-effects to the inputs.

To draw the Mandelbrot image, the ``mandel`` function is invoked on a set
of coordinates as shown below:

.. code-block:: python

    width = 2048
    height = 2048
    xs = numpy.linspace(-2.0, 1.0, width)
    ys = numpy.linspace(-1.0, 1.0, height)
    img = numpy.zeros((width, height), dtype=numpy.int8)
    for i in range(xs.size):
        for j in range(ys.size):
            img[i, j] = mandel(xs[i], ys[j])

Since ``mandel`` is pure and each assignment to ``img[i, j]`` is independent
of other iteration, the entire loop nest is data-independent.  The loop can be
executed concurrently by multithreads.

However, the CPython interpreter has a *global interpreter lock* (GIL) that
prevents the execution of Python instructions in parallel.  It guards against
race conditions from happening internally but serializes the execution of Python
instructions.  To take advantage of multicore processors, it is common to
rewrite the performance critical parts of a program in C so that the GIL can
be released for parallel execution.  However, writing complex algorithms
in C can be error-prone and time consuming.

Since Numba compiles Python functions into native machine code without any
dependency on the CPython API, the GIL can be released during the execution
of compiled functions.  The ``numba.jit`` decorator provides the ``nogil``
option to release of the GIL when the decorated function is called.
For example, the above loop-nest can be generalized into the following
fucntion:

.. code-block:: python

    @numba.jit(nogil=True)
    def mandel_tile(xs, ys, out):
        for i in range(xs.size):
            for j in range(ys.size):
                out[i, j] = mandel(xs[i], ys[j])
        return out

The ``mandel_tile`` will draw the Mandelbrot to ``out`` for the coordinates
given in ``xs`` and ``ys``.  The ``nogil`` option ensures the release of the
GIL when the function is called so that multiple threads can execute this
function in parallel.

To speedup the generation of the image, we can use the thread pool in
``concurrent.futures`` to execute above function for non-overlapping image
tiles.  For example:

.. code-block:: python

    npar = 2
    x_step = img.shape[0] // npar
    y_step = img.shape[1] // npar
    with concurrent.futures.ThreadPoolExecutor(4) as exe:
        futs = []
        # Submit job for each tile
        for pos_x in range(0, img.shape[0], x_step):
            for pos_y in range(0, img.shape[1], y_step):
                futs.append(exe.submit(mandel_tile, xs[pos_x : pos_x + x_step],
                                       ys[pos_y : pos_y + y_step], img1[pos_x:, pos_y:]))
        # Wait for all futures to complete
        for f in futs:
            f.result()

Using 4 threads and 4 tiles, a 2048x2048 image is generated in 43.5ms.  It is
4x faster than the serial execution, which takes 133ms.


.. WIP
    The above ``low_rank_approx`` function computes the low-rank approximation
    of any matrix ``x`` using the singular-value decomposition (SVD) and matrix
    multiplication (via ``dot``) routines from Numpy.  The majority of the
    computation time will be in these two routines.  In fact, simply applying
    ``jit`` on the function provides little speedup.  For a :math:`100\times50`
    input matrix, there is less than 10% speedup for the Numba compiled version.
    The reason is that Numba uses the same underlying BLAS routines for the linear
    algebra operations as NumPy.  However, it makes a bigger difference when
    the function is used many times on a batch of matrices in multiple threads.

    If we compare the following single-threaded version:

    .. code-block:: python

        xs = numpy.random.random((1000, 100, 50))
        zs = [low_rank_approx(x) for x in xs]

    and the multithreaded version using ``concurrent.futures``:

    .. code-block:: python

        xs = numpy.random.random((1000, 100, 50))
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exe:
            futs = exe.map(low_rank_approx, xs)
            zs = list(futs)

    The single-threaded version took 854ms verus 401ms for the multithreaded
    version, which uses 4 threads.  One may wonder why the 4-threads version is only
    twice as fast.  The reason is that the underlying BLAS, which is Intel MKL, is
    already multithreaded.  Each