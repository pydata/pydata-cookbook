Release the GIL for Parallel Execution
--------------------------------------

Some algorithms can be trivially parallelized to take advantage of the multicore
processors.  To identify subprograms that are potential candidates for
parallelism, look for *pure* functions and *data-independent* loops.
A function is *pure* if its outputs solely depends on its input arguments and
it does not have any side-effects that may mutate any global states.
For a loop to be *data independent*, each iteration has no dependency on each
other.  Therefore, the iterations can be completed in arbitrary order.

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
The function returns the depth of the iteration for the complex number to escape.
The function is *pure* given that it calculates its return value from the inputs
and it does not depend on any global variables.

To visualize the Mandelbrot set, the ``mandel`` function is invoked on a set
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

The loop nest calls ``mandel`` on each iteration and storing the depth as pixel
value for the image.
Since ``mandel`` is pure and each assignment to ``img[i, j]`` is independent,
the entire loop nest is data-independent.  The loop can be executed concurrently
by multithreads with arbitrary order.

However, the CPython interpreter has a *global interpreter lock* (GIL) that
prevents the execution of Python instructions in parallel.  It guards against
race conditions from happening internally but serializes the execution of
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

The output image is splitted into tiles.
In the first loop, tasks are submitted to the thread pool to compute on each
tile.  The call to ``exe.submit()`` enqueue the task on the thread pool and
returns before the task is completed.  It's return value is a future that is
used to track the progress of the computation.  The seecond loop waits for
all task to complete.  Since we use output arguments, the returned value by
each task is unnecessary and discarded.  Otherwise, the returned value can be
collected from ``f.result()``.

Using 4 threads and 4 tiles on a quardcore machine,
a 2048x2048 image can be  generated in 43.5ms.
It is 4x faster than the serial execution, which takes 133ms.

Speedup may not always be linear to the amount of parallelism.
Linear speedup is possible in the previous example due to the compute-bound
nature of the algorithm.  For each call to ``mandel``, only two doubles are
consumed but it can iterate up to 20 times and calling 6 floating-point
operations each time.  The program will run faster given more execution units.
On the other hand, a memory-bound program will be less likely to gain
linear speedup with multithreads.  In a memory-bound program, instruction
execution is stalled by pending memory requests.  More execution units will not
speedup the completion of memory requests.

Scientific applications that depends on linear algebra routines may not
see any speedup by using multiple threads due to the parallel implementation of
many underlying BLAS routines.  Nesting parallel code will just oversubscribe
the processor and increase the frequency of context-switching.  In the worst
case, the application performance can degrade.

Let's consider the following function:

.. code-block:: python

    @numba.jit(nogil=True)
    def low_rank_approx(x, k=10):
        u, s, v = numpy.linalg.svd(x)
        return numpy.dot(u[:, :k], numpy.dot(numpy.diag(s[:k]), v[:k, :]))


The above ``low_rank_approx`` function computes the low-rank approximation
of any matrix ``x`` using the singular-value decomposition (SVD) and matrix
multiplication (via ``dot``) routines from Numpy.  The majority of the
computation time will be in these two routines.  In fact, simply applying
``jit`` on the function provides little speedup.  For a :math:`800\times600`
input matrix, there is less than 10% speedup for the Numba compiled version.
The reason is that Numba uses the same underlying BLAS routines for the linear
algebra operations as NumPy.

If we compare the following single-threaded version:

.. code-block:: python

    xs = numpy.random.random((30, 800, 600))
    zs = [low_rank_approx(x) for x in xs]

and the 4-threaded version using ``concurrent.futures``:

.. code-block:: python

    xs = numpy.random.random((30, 800, 600))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as exe:
        futs = exe.map(low_rank_approx, xs)
        zs = list(futs)

The single-threaded version took 3.7s verus 4.1s for the 4-threaded
version.  Executing in multiple threads is harmful to the performance.
