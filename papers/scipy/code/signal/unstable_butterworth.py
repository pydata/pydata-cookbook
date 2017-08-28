from __future__ import division, print_function

import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt


b, a = butter(10, [0.04, 0.16], btype="bandpass")
x = np.ones(125)
y = lfilter(b, a, x)

plt.figure(figsize=(4.0, 2.0))
plt.plot(y)
plt.xlabel('Sample number')
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('unstable_butterworth.pdf')
