from __future__ import division, print_function

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
import matplotlib.pyplot as plt


n = 101
t = np.linspace(0, 1, n)
np.random.seed(123)
x = 0.45 + 0.1*np.random.randn(n)

sos = butter(8, 0.125, output='sos')

# Filter using the default initial conditions.
y = sosfilt(sos, x)

# Filter using the state for which the output
# is the constant x[:4].mean() as the initial
# condition.
zi = x[:4].mean() * sosfilt_zi(sos)
y2, zo = sosfilt(sos, x, zi=zi)

# Plot everything.
plt.figure(figsize=(4.0, 2.8))
plt.plot(t, x, alpha=0.75, linewidth=1, label='x')
plt.plot(t, y, label='y  (zero ICs)')
plt.plot(t, y2, label='y2 (mean(x[:4]) ICs)')

plt.legend(framealpha=1, shadow=True)
plt.grid(alpha=0.25)
plt.xlabel('t')
plt.title('Filter with different '
          'initial conditions',
          fontsize=10)
plt.tight_layout()
plt.savefig("initial_conditions.pdf")
