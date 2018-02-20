from __future__ import division, print_function

import numpy as np
from scipy.signal import firls, freqz
import matplotlib.pyplot as plt


numtaps = 43

fs = 200
f1 = 15
f2 = 30

bands =   np.array([0, f1, f1, f2, f2, 0.5*fs])
desired = np.array([1,  1,  1,  0,  0,      0])
taps1 = firls(numtaps, bands, desired, fs=fs)

w1, h1 = freqz(taps1, worN=8000)
w1 *= 0.5*fs/np.pi

wts = [100, .01, 1]
taps2 = firls(numtaps, bands, desired, weight=wts, fs=fs)

w2, h2 = freqz(taps2, worN=8000)
w2 *= 0.5*fs/np.pi


plt.figure(figsize=(4.0, 4.5))

plt.subplot(3, 1, 1)

for band, des in zip(bands.reshape(-1, 2), desired.reshape(-1, 2)):
    plt.plot(band, des, 'k', alpha=0.1, linewidth=4)
plt.plot(w1, np.abs(h1), alpha=0.9, label='uniform weight')
plt.plot(w2, np.abs(h2), alpha=0.9, label='weight:%s' % (wts,))

plt.ylim(-0.1, 1.1)
plt.ylabel('Gain')
ax = plt.gca()
xticks = np.r_[bands[::2], bands[-1]]
ax.set_xticks(xticks)
xticklabels = ['%g' % f for f in xticks]
ax.set_xticklabels(xticklabels)
plt.grid(alpha=0.25)
plt.legend(framealpha=1, shadow=True)
plt.title('Least Squares Filter Design', fontsize=10)

plt.subplot(3, 1, 2)

for band, des in zip(bands.reshape(-1, 2), desired.reshape(-1, 2)):
    plt.plot(band, des, 'k', alpha=0.1, linewidth=4)
plt.plot(w1, np.abs(h1), alpha=0.9)
plt.plot(w2, np.abs(h2), alpha=0.9)

ax = plt.gca()
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)

plt.ylim(0.985, 1.015)
plt.ylabel('Gain')
plt.xlim(0, 1.1*f1)

plt.grid(alpha=0.25)

plt.subplot(3, 1, 3)

for band, des in zip(bands.reshape(-1, 2), desired.reshape(-1, 2)):
    plt.plot(band, des, 'k', alpha=0.1, linewidth=4)
plt.plot(w1, np.abs(h1), alpha=0.9)
plt.plot(w2, np.abs(h2), alpha=0.9)

ax = plt.gca()
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)

plt.ylim(-0.002, 0.02)
plt.xlim(0.87*f2, 0.5*fs)
plt.grid(alpha=0.25)
plt.ylabel('Gain')

plt.xlabel('Frequency (Hz)')

plt.tight_layout()
plt.savefig('firls_example.pdf')
