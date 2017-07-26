import numpy as np
from scipy.signal import butter, sosfreqz, sosfilt
import matplotlib.pyplot as plt


sos = butter(10, [0.04, 0.16],
             btype="bandpass", output="sos")
w, h = sosfreqz(sos, worN=8000)

# Plot the magnitude of the frequency response.
plt.figure(figsize=(4.0, 2.0))
plt.plot(w/np.pi, np.abs(h))
plt.grid(alpha=0.25)
plt.xlabel('Normalized frequency')
plt.ylabel('Gain')
plt.tight_layout()
plt.savefig("sos_bandpass_response_freq.pdf")

# Plot the step response.
x = np.ones(200)
y = sosfilt(sos, x)

plt.figure(figsize=(4.0, 2.0))
plt.plot(y)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("sos_bandpass_response_step.pdf")
