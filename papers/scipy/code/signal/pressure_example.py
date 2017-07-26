from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sosfiltfilt, spectrogram
from butter import butter_lowpass, butter_lowpass_filtfilt


time, pressure = np.loadtxt('pressue.dat', skiprows=2,
                            delimiter=',', unpack=True)

t0 = time.min()
t1 = time.max()

fs = 50000

nperseg = 80
noverlap = nperseg - 4
window = 'hann'
#window = ('tukey', 0.5)
#window = ('kaiser', 12)

print("Spectrogram window length: %g ms (%d samples)" % (1000*nperseg/fs, nperseg))

f, t, Sxx = spectrogram(pressure, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
t += t0/1000

cutoff = 1250
pressure_filtered = butter_lowpass_filtfilt(pressure, cutoff, fs, order=8)

f, t, FSxx = spectrogram(pressure_filtered, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
t += t0/1000

tlo = t[0]*1000
thi = t[-1]*1000

spec_rel_thresh = 1e-9
x1 = np.log(Sxx.clip(spec_rel_thresh*Sxx.max(), Sxx.max()))
x2 = np.log(FSxx.clip(spec_rel_thresh*FSxx.max(), FSxx.max()))
vmin = min(x1.min(), x2.min())
vmax = max(x1.max(), x2.max())

linecolor = 'k'
#cmap = plt.cm.inferno
cmap = plt.cm.coolwarm
#linecolor = '#A06340'
#cmap = plt.cm.copper

plt.figure(figsize=(4.0, 4.5))
plt.subplot(211)
plt.plot(time, pressure, color=linecolor, alpha=0.8)
plt.xlim(tlo, thi)
plt.ylim(0, 10)
plt.ylabel("Pressure (MPa)")
plt.grid(alpha=0.25)
plt.subplot(212)

plt.pcolormesh(1000*t, f/1000, x1, cmap=cmap, vmin=vmin, vmax=vmax)
plt.xlim(tlo, thi)
plt.ylabel('Frequency (kHz)')
plt.xlabel('Time (ms)')
plt.tight_layout()
plt.savefig("pressure_example_input.pdf")

plt.figure(2, figsize=(4.0, 4.5))
plt.subplot(211)
plt.plot(time, pressure, color=linecolor, alpha=0.15)
plt.plot(time, pressure_filtered, color=linecolor, alpha=0.8)
plt.xlim(tlo, thi)
plt.ylim(0, 10)
plt.ylabel("Pressure (MPa)")
plt.grid(alpha=0.25)
plt.subplot(212)
plt.pcolormesh(1000*t, f/1000, x2, cmap=cmap, vmin=vmin, vmax=vmax)
plt.xlim(tlo, thi)
plt.ylabel('Frequency (kHz)')
plt.xlabel('Time (ms)')
plt.tight_layout()
plt.savefig("pressure_example_filtered.pdf")
