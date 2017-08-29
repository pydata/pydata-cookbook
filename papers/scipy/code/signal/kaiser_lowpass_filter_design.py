# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
from scipy.signal import kaiserord, firwin, freqz
import matplotlib.pyplot as plt


def kaiser_lowpass(delta_db, cutoff, width, fs):
    """
    Design a lowpass filter using the Kaiser window method.
    """
    # Convert to normalized frequencies
    nyq = 0.5*fs
    cutoff = cutoff / nyq
    width = width / nyq

    # Design the parameters for the Kaiser window FIR filter.
    numtaps, beta = kaiserord(delta_db, width)
    numtaps |= 1  # Ensure a Type I FIR filter.

    taps = firwin(numtaps, cutoff, window=('kaiser', beta), scale=False)

    return taps, beta


# User inputs...
# Values in Hz
fs = 1000.0
cutoff = 180.0
width = 30.0
deltap = 0.005
deltas = 0.002
delta = min(deltap, deltas)
stop_db = -20*np.log10(delta)

# Filter design...
taps, beta = kaiser_lowpass(stop_db, cutoff, width, fs)

print("Inputs")
print("------")
print("fs:", fs)
print("cutoff:", cutoff)
print("transition band width:", width)
print("delta:", delta, " (%.3f dB)" % stop_db)
print()
print("Kaiser design")
print("-------------")
print("numtaps:", len(taps))
print("beta: %.3f" % beta)

# Compute and plot the frequency response...
w, h = freqz(taps, worN=8000)
w *= 0.5*fs/np.pi

plt.figure(figsize=(4.0, 4.6))

plt.subplot(3, 1, 1)
plt.plot(w, 20*np.log10(np.abs(h)))
upper_ripple = 20*np.log10(1 + delta)
lower_ripple = 20*np.log10(1 - delta)
lower_trans = cutoff - 0.5*width
upper_trans = cutoff + 0.5*width

plt.plot([0, lower_trans], [upper_ripple, upper_ripple], 'r',
         linewidth=1, alpha=0.4)
plt.plot([0, lower_trans], [lower_ripple, lower_ripple], 'r',
         linewidth=1, alpha=0.4)
plt.plot([upper_trans, 0.5*fs], [-stop_db, -stop_db], 'r',
         linewidth=1, alpha=0.4)
plt.plot([lower_trans, lower_trans], [-stop_db, upper_ripple], color='r',
         linewidth=1, alpha=0.4)
plt.plot([upper_trans, upper_trans], [-stop_db, upper_ripple], color='r',
         linewidth=1, alpha=0.4)

plt.ylim(-1.8*stop_db, 10)
plt.ylabel('Gain (dB)')
plt.title('Kaiser Window Filter Design', fontsize=10)

plt.grid(alpha=0.25)

plt.subplot(3, 1, 2)
plt.plot(w, np.abs(h))
plt.plot([0, lower_trans], [1 + delta, 1 + delta], 'r',
         linewidth=1, alpha=0.4)
plt.plot([0, lower_trans], [1 - delta, 1 - delta], 'r',
         linewidth=1, alpha=0.4)
plt.plot([upper_trans, 1], [delta, delta], 'r', linewidth=1, alpha=0.4)
plt.plot([lower_trans, lower_trans], [delta, 1 + delta], color='r',
         linewidth=1, alpha=0.4)
plt.plot([upper_trans, upper_trans], [delta, 1 + delta], color='r',
         linewidth=1, alpha=0.4)

plt.ylim(1 - 1.5*delta, 1 + 1.5*delta)
plt.ylabel('Gain')
plt.xlim(0, cutoff)
plt.grid(alpha=0.25)

plt.subplot(3, 1, 3)
desired = w < cutoff
deviation = np.abs(np.abs(h) - desired)
deviation[(w >= cutoff-0.5*width) & (w <= cutoff + 0.5*width)] = np.nan
plt.plot(w, deviation)
plt.plot([0, 0.5*fs], [deltas, deltas], 'r', linewidth=1, alpha=0.4)
plt.ylabel(u'|A(ω) - D(ω)|')
plt.grid(alpha=0.25)

plt.xlabel('Frequency (Hz)')
plt.tight_layout()

plt.savefig('kaiser_lowpass_filter_design.pdf')
