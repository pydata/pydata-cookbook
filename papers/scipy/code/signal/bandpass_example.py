from __future__ import division, print_function

import numpy as np
from scipy.signal import sosfreqz
import matplotlib.pyplot as plt
from butter import butter_bandpass, butter_bandpass_filt
from make_data import make_data


# Sample rate and desired cutoff frequencies (in Hz).
fs = 4800.0
lowcut = 400.0
highcut = 1200.0

# Plot the frequency response for a few different orders.
plt.figure(figsize=(4.0, 4.0))
# First plot the desired ideal response as a green(ish) rectangle.
rect = plt.Rectangle((lowcut, 0), highcut - lowcut, 1.0,
                     facecolor="#60ff60",
                     edgecolor='k',
                     alpha=0.15)
plt.gca().add_patch(rect)
for order in [3, 6, 12]:
    sos = butter_bandpass(lowcut, highcut, fs, order)
    w, h = sosfreqz(sos, worN=2000)
    plt.plot((fs*0.5/np.pi)*w, abs(h), 'k',
             alpha=(order+1)/13, label="order = %d" % order)

plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
         'k--', alpha=0.6, linewidth=1, label=r'$\sqrt{2}/2$')
plt.xlim(0, 0.5*fs)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(alpha=0.5)
plt.legend(framealpha=1, shadow=True, loc='best')
plt.title("Amplitude response for\nButterworth bandpass filters", fontsize=10)
plt.text(430, 0.07, "lowcut: %4g Hz\nhighcut: %4g Hz" % (lowcut, highcut),
         fontsize=8)
plt.tight_layout()
plt.savefig("bandpass_example_response.pdf")

T = 0.03
t, x = make_data(T, fs)

y = butter_bandpass_filt(x, lowcut, highcut, fs, order=12)

plt.figure(figsize=(4.0, 2.8))
plt.plot(t, x, 'k', alpha=0.4, linewidth=1, label='Noisy signal')

plt.plot(t, y, 'C0', label='Filtered signal')
plt.xlabel('Time (seconds)')
plt.grid(alpha=0.5)
plt.axis('tight')
plt.xlim(0, T)
plt.legend(framealpha=1, shadow=True, loc='upper left')
plt.tight_layout()
plt.savefig("bandpass_example_signals.pdf")
