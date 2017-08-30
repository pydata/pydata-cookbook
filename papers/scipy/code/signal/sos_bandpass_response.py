from __future__ import division, print_function

import numpy as np
from scipy.signal import butter, sosfreqz, sosfilt
import matplotlib.pyplot as plt


sos = butter(10, [0.04, 0.16],
             btype="bandpass", output="sos")
w, h = sosfreqz(sos, worN=8000)

# Plot the magnitude and phase of the frequency response.
plt.figure(figsize=(4.0, 4.0))
plt.subplot(211)
plt.plot(w/np.pi, np.abs(h))
plt.grid(alpha=0.25)
plt.ylabel('Gain')
plt.subplot(212)
plt.plot(w/np.pi, np.angle(h))
yaxis = plt.gca().yaxis
yaxis.set_ticks([-np.pi, -0.5*np.pi, 0, 0.5*np.pi, np.pi])
yaxis.set_ticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
plt.xlabel('Normalized frequency')
plt.grid(alpha=0.25)
plt.ylabel('Phase')
plt.tight_layout()
plt.savefig("sos_bandpass_response_freq.pdf")

# Plot the step response.
x = np.ones(200)
y = sosfilt(sos, x)

plt.figure(figsize=(4.0, 2.0))
plt.plot(y)
plt.grid(alpha=0.25)
plt.xlabel('Sample number')
plt.tight_layout()
plt.savefig("sos_bandpass_response_step.pdf")
