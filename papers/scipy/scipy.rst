:author: Evgeni Burovski
:email: evgeny.burovskiy@gmail.com
:institution: Higher School of Economics, Russia

:author: Ralf Gommers
:email: ralf.gommers@gmail.com
:institution:

:author: Warren Weckesser
:email: warren.weckesser@gmail.com

..
    Typography question: "lowpass", "low-pass" or "low pass"?
    I (WW) will follow the convention used in the two books that I happen
    to have handy (Oppenheim and Schafer, "Discrete-Time Signal Processing",
    and Richard G. Lyons, "Understanding Digital Signal Processing"), and will
    use "lowpass", "highpass" and "bandpass" when discussing filters.  I don't
    really have a strong preference, but it will save some copy-editing later
    if we agree on the convention now.

..
    Some LaTeX typography comments:
    I (WW) find LaTeX's default size for subscripts is too big.  That why
    I write, for example, `a_{_N}` instead of just `a_N`.  If you leave it
    as `a_N`, then in a formula such as `a_N z`, the N is practically the
    same size as and side-by-side with the z.  Using `a_{_N}` makes it
    very clear that N is a subscript of a.

-----
SciPy
-----

.. class:: abstract

The SciPy_ library is one of the core packages of the PyData stack.  It
includes modules for statistics, optimization, interpolation, integration,
linear algebra, Fourier transforms, signal and image processing, ODE solvers,
special functions, sparse matrices, and more.


.. _SciPy: http://scipy.org/scipylib/index.html

.. class:: keywords

algorithms, optimization, statistics, linear algebra, signal processing,
sparse matrix, interpolation, numerical integration, special functions

.. contents::

Introduction
============

[TO DO...]

Related projects
----------------

* ``statsmodels.tsa``: Time series analysis module in ``statsmodels``
* others...

Filtering with ``scipy.signal``
===============================

In this section we show how SciPy can be used for basic linear filtering
of time series data.

There are two main classes of linear filters: *finite impulse response* (FIR)
filters, and *infinite impulse response* (IIR) filters.  SciPy provides
tools for the design and analysis of both types, but in this chapter, we
will focus on IIR filters.

An IIR filter can be written as a linear recurrence relation, in which the
output ``y[n]`` is a linear combination of ``x[n]``, the `M` previous
values of ``x`` and the `N` previous values of ``y``:

.. math::
   :label: eq-filter-recurrence

   a_{_0} y_{_n} = \sum_{i=0}^{M} b_{_i}x_{_{n-i}} -
                  \sum_{i=1}^{N} a_{_i} y_{_{n-N}} 

(See, for example, [OppenheimSchafer]_, Chapter 6.  Note, however,
that SciPy uses a different sign convention for ``a[1], ..., a[N]``.)

The rational transfer function describing this filter in the
z-transform domain is

.. math::

    Y(z) = \frac{b_{_0} + b_{_1} z^{-1} + \cdots + b_{_M} z^{-M}}
                {a_{_0} + a_{_1} z^{-1} + \cdots + a_{_N} z^{-N}} X(z)

The functions in SciPy that create filters generally set
:math:`a_{_0} = 1`.

IIR filter representation
-------------------------

In this section, we discuss three representations of a linear filter:

* transfer function
* zeros, poles, gain (ZPK)
* second order sections (SOS)

SciPy also provides a state space representation,
but we won't discuss that format here.

The *transfer function* representation of
a filter in SciPy is two one-dimensional arrays, conventionally
called ``b`` and ``a``, that hold the coefficients of the polynomials
in the numerator and denominator, respectively, of the transfer function
H(z).

For example, we can use the function ``scipy.signal.butter`` to
create a Butterworth lowpass filter of order 6 with a normalized
cutoff frequeny of 1/8 the Nyquist rate.  The default representation
created by ``butter`` is the transfer function, so we can use
``butter(6, 0.125)``.
(We use ``numpy.set_printions(precision=3, linewidth=50)``
in all interactive Python sessions.)::

    >>> from scipy.signal import butter
    >>> b, a = butter(6, 0.125)
    >>> b
    array([  2.883e-05,   1.730e-04,   4.324e-04,
             5.765e-04,   4.324e-04,   1.730e-04,
             2.883e-05])
    >>> a
    array([ 1.   , -4.485,  8.529, -8.779,  5.148,
           -1.628,  0.217])

The *ZPK* representation consists of a tuple containing three
items, ``(z, p, k)``.  The first two items, ``z`` and ``p``, are
one-dimensional arrays containing the zeros and poles, respectively,
of the transfer function.  The third item, ``k``, is a scalar that holds
the overall gain of the filter.

We can tell ``butter`` to create a filter using the ZPK representation
by using the argument ``output="zpk"``::

    >>> z, p, k = butter(6, 0.125, output='zpk')
    >>> z
    array([-1., -1., -1., -1., -1., -1.])
    >>> p
    array([ 0.841+0.336j,  0.727+0.213j,
            0.675+0.072j,  0.675-0.072j,
            0.727-0.213j,  0.841-0.336j])
    >>> k
    2.8825891944002783e-05

In the *second order sections (SOS)* representation, the filter is represented
using one or more second order filters (also known as "biquads").
The SOS representation is implemented as an array with shape (n, 6),
where each row holds the coefficients of a second order transfer function.
The first three items in a row are the coefficients of the numerator of the
biquad's transfer function, and the second three items are the coefficients
of the denominator.

Here we create a Butterworth filter using the SOS representation::

    >>> sos = butter(6, 0.125, output="sos")
    >>> sos
    array([[  2.883e-05,   5.765e-05,   2.883e-05,
              1.000e+00,  -1.349e+00,   4.602e-01],
           [  1.000e+00,   2.000e+00,   1.000e+00,
              1.000e+00,  -1.454e+00,   5.741e-01],
           [  1.000e+00,   2.000e+00,   1.000e+00,
              1.000e+00,  -1.681e+00,   8.198e-01]])

The ``signal`` module provides a collection of functions for
converting one representation to another.  For example, ``zpk2sos``
converts from the ZPK representation to the SOS representation::

    >>> from scipy.signal import zpk2sos
    >>> zpk2sos(z, p, k) 
    array([[  2.883e-05,   5.765e-05,   2.883e-05,
              1.000e+00,  -1.349e+00,   4.602e-01],
           [  1.000e+00,   2.000e+00,   1.000e+00,
              1.000e+00,  -1.454e+00,   5.741e-01],
           [  1.000e+00,   2.000e+00,   1.000e+00,
              1.000e+00,  -1.681e+00,   8.198e-01]])

The representation of a filter as a transfer function with coefficients
``(b, a)`` is convenient and of theoretical importance, but the use
of polynomials of even moderately high order makes the representation
sensitive to numerical errors.  Problems can arise when designing a
filter of high order, or a filter with very narrow pass or stop bands.

Consider, for example, the design of a Butterworth bandpass filter
with order 10 with normalized pass band cutoff frequencies of 0.04
and 0.16.::

    >>> b, a = butter(10, [0.04, 0.16], btype="bandpass")

We can compute the step response of this filter by applying it to
an array of ones.  (We're just spot-checking our calculation, so
we won't bother with the usual plot annotations.)::

    >>> x = np.ones(125)
    >>> y = lfilter(b, a, x)
    >>> plt.plot(y)

The plot is shown in Figure :ref:`fig-unstable-butterworth`.
Clearly something is going wrong.

.. figure:: figs/unstable_butterworth.pdf

    Step response of the Butterworth bandpass filter of order 10 created
    using the transfer function representation.  Apparently the
    filter is unstable--something has gone wrong with this representation.
    :label:`fig-unstable-butterworth`

We can try to determine the problem by checking the poles
of the filter::

    >>> z, p, k = tf2zpk(b, a)
    >>> np.abs(p)
    array([ 0.955,  0.955,  1.093,  1.093,  1.101,
            1.052,  1.052,  0.879,  0.879,  0.969,
            0.969,  0.836,  0.836,  0.788,  0.788,
            0.744,  0.744,  0.725,  0.725,  0.723])

The filter should have all poles inside the unit
circle in the complex plane, but in this case five of the poles
have magnitude greater than 1.
This indicates a problem, which could be in the
result returned by ``butter``, or in the conversion done
by ``tf2zpk``.  The plot shown in Figure :ref:`fig-unstable-butterworth`
makes clear that *something* is wrong with the coefficients in
``b`` and ``a``.

If we request the ZPK representation from ``butter``,
the filter is stable::

    >>> z, p, k = butter(10, [0.04, 0.16],
    ...                  btype="bandpass", output="zpk")
    >>> np.abs(p) 
    array([ 0.988,  0.964,  0.936,  0.903,  0.854,
            0.854,  0.903,  0.936,  0.964,  0.988,
            0.955,  0.877,  0.818,  0.788,  0.8  ,
            0.8  ,  0.788,  0.818,  0.877,  0.955])

SciPy does not provide a function that applies a filter
directly using the ZPK format.  We could convert the values
``(z, p, k)`` to SOS, but in that case, we might as well create
the filter in SOS format at the start::

    >>> sos = butter(10, [0.04, 0.16],
    ...              btype="bandpass", output="sos")

Let's do the same check that we did with the transfer function
representation: convert to ZPK, and check the magnitudes of
the poles::

    >>> z, p, k = sos2zpk(sos)
    >>> np.abs(p)
    array([ 0.788,  0.788,  0.8  ,  0.8  ,  0.818,
            0.818,  0.854,  0.854,  0.877,  0.877,
            0.903,  0.903,  0.936,  0.936,  0.955,
            0.955,  0.964,  0.964,  0.988,  0.988])

We obtain the same magnitudes of the poles as when we created
the ZPK representation directly in ``butter``.  We can check
the frequency response using ``scipy.signal.sosfreqz``::

    >>> w, h = sosfreqz(sos, worN=8000)
    >>> plt.plot(w/np.pi, np.abs(h))
    [<matplotlib.lines.Line2D at 0x109ae9550>]
    >>> plt.grid(alpha=0.25)
    >>> plt.xlabel('Normalized frequency')
    >>> plt.ylabel('Gain')

The plot is shown in Figure :ref:`fig-sos-bandpass-response-freq`.

.. figure:: figs/bandpass_freq_response.pdf

    Frequency response of the Butterworth bandpass filter with
    order 10 and normalized cutoff frequencies 0.04 and 0.16.
    :label:`fig-sos-bandpass-response-freq`

As above, we compute the step response by filtering an array of ones::

    >>> x = np.ones(200)
    >>> y = sosfilt(sos, x)
    >>> plt.plot(y)
    >>> plt.grid(alpha=0.25)

The plot is shown in Figure :ref:`fig-sos-bandpass-response-step`.

.. figure:: figs/sos_bandpass_response_step.pdf

    Step response of the Butterworth bandpass filter with
    order 10 and normalized cutoff frequencies 0.04 and 0.16.
    :label:`fig-sos-bandpass-response-step`

With the SOS representation, the filter behaves as expected.

In the remaing examples of filtering, we will use only the
SOS representation.

Lowpass filter
--------------

Figure :ref:`fig-pressure-example-input` shows a times series containing
pressure measurements. At some point in the interval 20 < t < 22,
an event occurs in which the pressure jumps and begins oscillating
around a "center".  The center of the oscillation decreases and
appears to level off.

.. figure:: figs/pressure_example_input.pdf

   *Top*: Pressure, for the interval 15 < t < 35 (milliseconds).
   *Bottom*: Spectrogram of the pressure time series (generated using a
   window size of 1.6 milliseconds).
   :label:`fig-pressure-example-input`

We are not interested in the oscillations, but we are interested in the mean
value around which the signal is oscillating.

To preserve the slowly varying behavior while eliminating the high frequency
oscillations, we'll apply a low-pass filter.  To apply the filter, we can
use either ``sosfilt`` or ``sosfiltfilt`` from ``scipy.signal``.
The function ``sosfiltfilt`` is a forward-backward filter--it applies the
filter twice, once forward and once backward.  This effectively doubles the
order of the filter, and results in zero phase shift.
Because we are interesting in the "event" that occurs in 20 < t < 22,
it is important to preserve the phase characteristics of the signal, so
we use ``sosfiltfilt``.

The following code snippet defines two convenience functions.  These
functions allow us to specify the sampling frequency and the lowpass
cutoff frequency in whatever units are convenient.  They take care of
scaling the values to the units expected by ``scipy.signal.butter``.


.. code-block:: python

    from scipy.signal import butter, sosfiltfilt

    def butter_lowpass(cutoff, fs, order):
        normal_cutoff = cutoff / (0.5*fs)
        sos = butter(order, normal_cutoff,
                     btype='low', output='sos')
        return sos

    def butter_lowpass_filtfilt(data, cutoff, fs,
                                order):
        sos = butter_lowpass(cutoff, fs, order=order,
                              output='sos')
        y = sosfiltfilt(sos, data)
        return y

The results of filtering the data using ``sosfiltfilt`` are shown in
Figure :ref:`fig-pressure-example-filtered`.

.. figure:: figs/pressure_example_filtered.pdf

   *Top*: Filtered pressure, for the interval 15 < t < 35 (milliseconds).
   The light gray curve is the unfiltered data.
   *Bottom*: Spectrogram of the filtered time series (generated using a
   window size of 1.6 milliseconds).
   :label:`fig-pressure-example-filtered`

Initializing a lowpass filter
-----------------------------

By default, the initial state of an IIR filter as implemented in
``lfilter`` or ``sosfilt`` is all zero.  If the input signal does not
start with values that are zero, there will be a transient during which
the filter's internal state "catches up" with the input signal.

Here is an example.  The script generates the plot shown in
Figure :ref:`fig-initial-conditions`.

.. code-block:: python

    import numpy as np
    from scipy.signal import butter, sosfilt, sosfilt_zi
    import matplotlib.pyplot as plt

    n = 101
    t = np.linspace(0, 1, n)
    np.random.seed(123)
    x = 0.45 + 0.1*np.random.randn(n)

    sos = butter(8, 0.125, output='sos')

    # Filter using the default initial conditions.
    y = sosfilt(sos, x)

    # Filter using the state for which the output
    # is the constant x[:4].mean() as the initial
    # condition.
    zi = x[:4].mean() * sosfilt_zi(sos)
    y2, zo = sosfilt(sos, x, zi=zi)

    # Plot everything.
    plt.plot(t, x, alpha=0.75, linewidth=1, label='x')
    plt.plot(t, y, label='y  (zero ICs)')
    plt.plot(t, y2, label='y2 (mean(x[:4]) ICs)')

    plt.legend(framealpha=1, shadow=True)
    plt.grid(alpha=0.25)
    plt.xlabel('t')
    plt.title('Filter with different '
              'initial conditions')
    plt.show()

By setting ``zi=x[:4].mean() * sosfilt_zi(sos)``, we are, in effect,
making the filter start out as if it had been filtering the constant
``x[:4].mean()`` for a long time.  There is still a transient associated
with this assumption, but it is usually not as objectionable as the
transient associated with zero initial conditions.

.. figure:: figs/initial_conditions.pdf
    
   A demonstration of two different sets of initial conditions for
   a lowpass filter.  The orange curve shows a filter with zero initial
   conditions.  The green curves show a filter that was initialized
   with a state associated with the first four values of the input ``x``.
   :label:`fig-initial-conditions`

This initialization is usually not needed for a bandpass
or highpass filter.  Also, the forward-backward filters implemented
in ``filtfilt`` and ``sosfiltfilt`` already have options for controlling
the initial conditions of the forward and backward passes.

Bandpass filter
---------------

In this example, we will use synthetic data to demonstrate a
bandpass filter.  We have 0.03 seconds of data sampled at
4800 Hz.  We want to apply a bandpass filter to remove frequencies
below 400 Hz or above 1200 Hz.

.. code-block:: python

    from scipy.signal import butter, sosfilt

    def butter_bandpass(lowcut, highcut, fs, order):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], btype='band',
                     output='sos')
        return sos

    def butter_bandpass_filt(data, lowcut, highcut,
                             fs, order):
        sos = butter_bandpass(lowcut, highcut, fs,
                              order)
        y = sosfilt(sos, data)
        return y

First, we'll take a look at the frequency response of the Butterworth
bandpass filter with order 3, 6, and 12.  The code that generates
Figure :ref:`fig-bandpass-example-response` demonstrates the use of
``scipy.signal.sosfreqz``:

.. code-block:: python

    for order in [3, 6, 12]:
        sos = butter_bandpass(lowcut, highcut, fs, order)
        w, h = sosfreqz(sos, worN=2000)
        plt.plot((fs*0.5/np.pi)*w, abs(h), 'k',
                 alpha=(order+1)/13,
                 label="order = %d" % order)

.. figure:: figs/bandpass_example_response.pdf

    Amplitude response for a Butterworth bandpass filter
    with several different orders.
    :label:`fig-bandpass-example-response`

Figure :ref:`fig-bandpass-example-signals` shows the input signal and
the filtered signal.  The order 12 bandpass Butterworth filter
was used.  The plot shows the input signal `x`; the filtered signal
was generated with

.. code-block:: python

    y = butter_bandpass_filt(x, lowcut, highcut, fs,
                             order=12)

where ``fs = 4800``, ``lowcut = 400`` and ``highcut = 1200``.

.. figure:: figs/bandpass_example_signals.pdf

    Original noisy signal and the filtered signal.
    The order 12 Butterworth bandpass filter shown in
    Figure :ref:`fig-bandpass-example-response` was used.
    :label:`fig-bandpass-example-signals`

Filtering a long signal in batches
----------------------------------

We will again use synthetic data generated by the same function
used in the previous example, but for a longer time interval.

This example shows how the state of the IIR filter can be saved
and restored, so a filter can be applied to a long signal in batches.

A pattern that can be used to filter an input signal ``x`` in
batches is shown in the following code.  The filtered signal
is stored in ``y``.  The array ``sos`` contains the filter
in SOS format, and is presumed to have already been created.
 
.. code-block:: python

    batch_size = N  # Number of samples per batch

    # Array of initial conditions for the SOS filter.
    z = np.zeros((sos.shape[0], 2))

    # Preallocate space for the filtered signal.
    y = np.empty_like(x)

    start = 0
    while start < len(x):
        stop = min(start + batch_size, len(x))
        y[start:stop], z = sosfilt(sos, x[start:stop],
                                   zi=z)
        start = stop

In this code, the next batch of input is fetched
by simply indexing ``x[start:stop]``, and the filtered
batch is saved by assigning it to ``y[start:stop]``.
In a more realistic batch processing system, the
input might be fetched from a file, or directly
from an instrument, and the output might be written
to another file, or handed off to another process
as part of a batch processing pipeline.

.. figure:: figs/bandpass_batch_example.pdf

    Original noisy signal and the filtered signal.
    The order 12 Butterworth bandpass filter shown in
    Figure :ref:`fig-bandpass-example-response` was used.
    The signal was filtered in batches of size 72 samples
    (0.015 seconds).  The alternating light and dark blue
    colors of the filtered signal indicate batches that
    were processed in separate calls to ``sosfilt``.
    :label:`fig-bandpass-batch-example`

Solving linear recurrence relations
-----------------------------------

Variations of the question::

        How do I speed up the following calculation?

        y[i+1] = alpha*y[i] + c*x[i]

often arise on mailing lists and online forums.  Sometimes more
terms such as ``beta*y[i-1]`` or ``d*x[i-1]`` are included on the right.
These recurrence relations show up in, for example, GARCH models
and other linear stochastic models.
Such a calculation can be written in the form of eqn.
(:ref:`eq-filter-recurrence`), so a solution can be computed
using ``lfilter``.

Here's an example that is similar to several questions that
have appeared on the programming Q&A website ``stackoverflow.com``.
The one-dimensional array  ``h`` is an input, and ``alpha``, ``beta`` and
``gamma`` are constants::

    y = np.empty(len(h))
    y[0] = alpha
    for i in np.arange(1, len(h)):
        y[i] = alpha + beta*y[i-1] + gamma*h[i-1]

To use ``lfilter`` to solve the problem, we have to translate
the linear recurrence::

    y[i] = alpha + beta*y[i-1] + gamma*h[i-1]

into the form of eqn. (:ref:`eq-filter-recurrence`), which will give us the
coefficients ``b`` and ``a`` of the transfer function.  Define::

    x[i] = alpha + gamma*h[i]

so the recurrence relation is::

    y[i] = x[i-1] + beta*y[i-1]

Compare this to eqn. (:ref:`eq-filter-recurrence`); 
we see that :math:`a_{_0} = 1`, :math:`a_{_1} = -\rm{beta}`,
:math:`b_{_0} = 0` and :math:`b_{_1} = 1`.
So we have our transfer function coefficients::

    b = [0, 1]
    a = [1, -beta]

We also have to ensure that the initial condition is set correctly to
reproduce the desired calculation.
We want the initial condition to be set as if we had values ``x[-1]``
and ``y[-1]``, and ``y[0]`` is computed using the recurrence relation.
Given the above recurrence relation, the formula for ``y[0]`` is::

    y[0] = x[-1] + beta*y[-1]

We want ``y[0]`` to be ``alpha``, so we'll set ``y[-1] = 0`` and
``x[-1] = alpha``.  To create initial conditions for ``lfilter``
that will set up the filter to act like it had just operated on
those previous values, we use ``scipy.signal.lfiltic``::

    zi = lfiltic(b, a, y=[0], x=[alpha])

The ``y`` and ``x`` arguments are the "previous" values that will
be used to set the initial conditions.  In general, one sets
``y=[y[-1], y[-2], ..]`` and ``x=[x[-1], x[-2], ...]``, giving as
many values as needed to determine the initial condition for
``lfilter``.  In this example, we have just one previous value
for ``y`` and ``x``.

Putting it all together, here is the code using ``lfilter`` that
replaces the for-loop shown above::

    b = [0, 1]
    a = [1, -beta]
    zi = lfiltic(b, a, y=[0], x=[alpha])
    y, zo = lfilter(b, a, alpha + gamma*h, zi=zi)



References
==========
.. [OppenheimSchafer]
    Alan V. Oppenheim, Ronald W. Schafer.
    *Discrete-Time Signal Processing*,
    Pearson Higher Education, Inc., Upper Saddle River,
    New Jersey (2010)
