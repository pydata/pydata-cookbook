# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from scipy.signal import remez, freqz
import matplotlib.pyplot as plt


def bellanger_estimate(deltap, deltas, width, fs):
    """
    Estimate the number of taps required for the given filter specifications.
    """
    n = (-2/3)*np.log10(10*deltap*deltas)*fs/width
    n = int(np.ceil(n))
    return n


def remez_lowpass(deltap, deltas, cutoff, width, fs):

    numtaps = bellanger_estimate(deltap, deltas, width, fs)
    numtaps |= 1  # Bitwise OR with 1 to ensure an odd number of taps.
    trans_lo = cutoff - 0.5*width
    trans_hi = cutoff + 0.5*width
    taps = remez(numtaps,
                 bands=[0, trans_lo, trans_hi, 0.5*fs],
                 desired=[1, 0],
                 weight=[1/deltap, 1/deltas],
                 fs=fs)
    return taps


#---------------------------------------
# User inputs...

# Frequency values in Hz
fs = 1000.0
cutoff = 180.0
width = 30.0
# Desired pass band ripple and stop band attenuation
deltap = 0.005
deltas = 0.002

print(u"Pass band:  1 ± %g ([%.3g, %.3g] dB)" %
      (deltap, 20*np.log10(1 - deltap), 20*np.log10(1 + deltap)))
print("Stop band rejection: %g (%.3g dB)" % (deltas, -20*np.log10(deltas),))

#---------------------------------------
# Design the filter...

taps = remez_lowpass(deltap, deltas, cutoff, width, fs)

#----------------------------------------
# Plot the frequency response...

upper_ripple_db = 20*np.log10(1 + deltap)
lower_ripple_db = 20*np.log10(1 - deltap)
stop_db = -20*np.log10(deltas)

print("Inputs")
print("------")
print("fs:", fs)
print("cutoff:", cutoff)
print("transition band width:", width)
print("deltap:", deltap, " (%.3f dB)" % (-20*np.log10(deltap),))
print("deltas:", deltas, " (%.3f dB)" % (-20*np.log10(deltas),))
print()
print("Design")
print("------")
print("numtaps:", len(taps))

w, h = freqz(taps, worN=8000)
w *= 0.5*fs/np.pi

cutoff_lower_trans = cutoff - 0.5*width
cutoff_upper_trans = cutoff + 0.5*width


plt.figure(figsize=(4.0, 4.6))
plt.subplot(3, 1, 1)

plt.plot(w, 20*np.log10(np.abs(h)))

plt.plot([0, cutoff_lower_trans], [upper_ripple_db, upper_ripple_db], 'r',
         alpha=0.4)
plt.plot([0, cutoff_lower_trans], [lower_ripple_db, lower_ripple_db], 'r',
         alpha=0.4)

plt.plot([cutoff_upper_trans, 0.5*fs], [-stop_db, -stop_db], 'r', alpha=0.4)

plt.axvline(cutoff_lower_trans, color='k', alpha=0.4, linewidth=1)
plt.axvline(cutoff_upper_trans, color='k', alpha=0.4, linewidth=1)

widthstr = '%g Hz' % width
if cutoff < 0.25*fs:
    lefttext = ''
    righttext = widthstr
else:
    lefttext = widthstr
    righttext = ''
plt.annotate(righttext, (cutoff_upper_trans, -0.5*stop_db),
             xytext=(18, 0),
             textcoords='offset points',
             va='center', ha='left',
             arrowprops=dict(arrowstyle='->'))
plt.annotate(lefttext, (cutoff_lower_trans, -0.5*stop_db),
             xytext=(-18, 0),
             textcoords='offset points',
             va='center', ha='right',
             arrowprops=dict(arrowstyle='->'))


plt.ylim(-1.25*stop_db, 10)
plt.grid(alpha=0.25)
plt.ylabel('Gain (dB)')
plt.title("Lowpass Filter\nOptimal Remez Design",
          fontsize=10)


plt.subplot(3, 1, 2)

plt.plot(w, np.abs(h))

plt.plot([0, cutoff_lower_trans], [1 + deltap, 1 + deltap], 'r', alpha=0.4)
plt.plot([0, cutoff_lower_trans], [1 - deltap, 1 - deltap], 'r', alpha=0.4)

plt.plot([cutoff_upper_trans, 0.5*fs], [deltas, deltas], 'r', alpha=0.4)

plt.axvline(cutoff_lower_trans, color='k', alpha=0.4, linewidth=1)
plt.axvline(cutoff_upper_trans, color='k', alpha=0.4, linewidth=1)

plt.xlim(0, (cutoff + 0.6*width))
plt.ylim(1 - 1.6*deltap, 1 + 1.6*deltap)

plt.ylabel('Gain')

plt.grid(alpha=0.25)

plt.subplot(3, 1, 3)
desired = w < cutoff
deviation = np.abs(np.abs(h) - desired)
deviation[(w >= cutoff-0.5*width) & (w <= cutoff + 0.5*width)] = np.nan
plt.plot(w, deviation)
plt.plot([0, cutoff - 0.5*width], [deltap, deltap], 'r',
         linewidth=1, alpha=0.4)
plt.plot([cutoff + 0.5*width, 0.5*fs], [deltas, deltas], 'r',
         linewidth=1, alpha=0.4)
plt.ylabel(u'|A(ω) - D(ω)|')
plt.grid(alpha=0.25)

plt.xlabel('Frequency (Hz)')

plt.tight_layout()
plt.savefig("opt_lowpass.pdf")
