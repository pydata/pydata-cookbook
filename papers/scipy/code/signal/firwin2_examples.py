# -*- coding: utf-8 -*-
from __future__ import division, print_function

import numpy as np
from scipy.signal import firwin2, freqz, boxcar, hamming, kaiser
import matplotlib.pyplot as plt


fs = 2000
freqs = [0, 48,  60, 72, 150, 175, 1000]
gains = [1,  1, 0.1,  1,   1,   0,    0]

numtaps = 185

taps_none = firwin2(numtaps, freqs, gains, fs=fs, window=None)
taps_h = firwin2(numtaps, freqs, gains, fs=fs)

beta = 2.70
taps_k = firwin2(numtaps, freqs, gains, fs=fs, window=('kaiser', beta))

w_none, h_none = freqz(taps_none, 1, worN=2000)
w_h, h_h = freqz(taps_h, 1, worN=2000)
w_k, h_k = freqz(taps_k, 1, worN=2000)

plt.figure(figsize=(4.0, 2.8))

win_boxcar = boxcar(numtaps)
win_hamming = hamming(numtaps)
win_kaiser = kaiser(numtaps, beta)

plt.plot(win_hamming, label='Hamming')
plt.plot(win_kaiser, label='Kaiser, $\\beta$=%.2f' % beta)
plt.plot(win_boxcar, label='rectangular')
plt.xticks([0, (numtaps - 1)//2, numtaps - 1])
plt.xlabel('Sample number')
plt.ylim(0, 1.05)
plt.grid(alpha=0.25)
plt.title("Window functions", fontsize=10)
plt.legend(framealpha=1, shadow=True)
plt.tight_layout()

plt.savefig("firwin2_examples_windows.pdf")

plt.figure(figsize=(4.0, 3.5))
plt.plot(freqs, gains, 'k--', alpha=0.5, linewidth=1, label='ideal')

plt.plot(0.5*fs*w_h/np.pi, np.abs(h_h), label='Hamming')
plt.plot(0.5*fs*w_k/np.pi, np.abs(h_k), label='Kaiser, $\\beta$=%.2f' % beta)
plt.plot(0.5*fs*w_none/np.pi, np.abs(h_none), label='rectangular')
plt.xlim(0, 210)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.legend(framealpha=1, shadow=True, loc='center right')
plt.grid(alpha=0.25)

plt.text(9, .08, "%d taps" % numtaps, fontsize=9)

plt.title('Filters designed with the window method', fontsize=10)
plt.tight_layout()

plt.savefig("firwin2_examples.pdf")
