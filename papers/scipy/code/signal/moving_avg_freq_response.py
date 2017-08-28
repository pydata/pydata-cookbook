from __future__ import division, print_function

import numpy as np
from scipy.signal import freqz
import matplotlib.pyplot as plt


plt.figure(figsize=(4.0, 3.0))

for n in [3, 7, 21]:
    taps = np.full(n, fill_value=1.0/n)
    w, h = freqz(taps, worN=2000)
    plt.plot(w, abs(h), label="n = %d" % n)

plt.xlim(0, np.pi)
plt.xlabel('Frequency (radians/sample)')
plt.ylabel('Gain')
xaxis = plt.gca().xaxis
xaxis.set_ticks([0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi])
xaxis.set_ticklabels(['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$',
                      r'$\frac{3\pi}{4}$', r'$\pi$'])
plt.grid(alpha=0.5)
plt.legend(framealpha=1, shadow=True, loc='best')
plt.title("Amplitude Response for\nMoving Average Filter", fontsize=10)
plt.tight_layout()
plt.savefig("moving_avg_freq_response.pdf")
