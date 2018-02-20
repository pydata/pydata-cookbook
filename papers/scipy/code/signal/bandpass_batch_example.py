from __future__ import division, print_function

import numpy as np
from scipy.signal import sosfilt
import matplotlib.pyplot as plt
from butter import butter_bandpass
from make_data import make_data


# Sample rate and desired cutoff frequencies (in Hz).
fs = 4800.0
lowcut = 400.0
highcut = 1200.0

T = 0.06
t, x = make_data(T, fs)

sos = butter_bandpass(lowcut, highcut, fs, 12)

batch_size = 72

# Array of initial conditions for the SOS filter.
z = np.zeros((sos.shape[0], 2))

# Preallocate space for the filtered signal.
y = np.empty_like(x)

start = 0
while start < len(x):
    stop = min(start + batch_size, len(x))
    y[start:stop], z = sosfilt(sos, x[start:stop], zi=z)
    start = stop


plt.figure(figsize=(4.0, 3.2))

plt.plot(t, x, 'k', alpha=0.4, linewidth=1, label='Noisy signal')

start = 0
alpha = 0.5
while start < len(x):
    stop = min(start + batch_size, len(x))
    if start == batch_size:
        label = 'Filtered signal'
    else:
        label = None
    plt.plot(t[start:stop+1], y[start:stop+1], 'C0', alpha=alpha, label=label)
    alpha = 1.5 - alpha
    start = stop

plt.xlabel('Time (seconds)')
plt.grid(alpha=0.5)
plt.axis('tight')
plt.xlim(0, T)
plt.legend(framealpha=1, shadow=True, loc='upper left')
plt.tight_layout()
plt.savefig("bandpass_batch_example.pdf")
