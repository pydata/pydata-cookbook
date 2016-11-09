import numba
import numpy
import collections
import pprint


@numba.jit
def assign_bin(val):
    limit = 32
    while limit < 2**12:
        if val < limit:
            return limit
        limit *= 2
    return limit


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


arr = numpy.random.random(10)
binning(arr)

binning.inspect_types()
