from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from butter import butter_lowpass_filtfilt


time, pressure = np.loadtxt('pressure.dat', skiprows=2,
                            delimiter=',', unpack=True)

t0 = time.min()
t1 = time.max()

fs = 50000

nperseg = 80
noverlap = nperseg - 4

print("Spectrogram window length: %g ms (%d samples)" %
      (1000*nperseg/fs, nperseg))

f, t, spec = spectrogram(pressure, fs=fs, nperseg=nperseg, noverlap=noverlap,
                         window='hann')
t += t0/1000

cutoff = 1250
pressure_filtered = butter_lowpass_filtfilt(pressure, cutoff, fs, order=8)

f, t, filteredspec = spectrogram(pressure_filtered, fs=fs, nperseg=nperseg,
                                 noverlap=noverlap, window='hann')
t += t0/1000

tlo = t[0]*1000
thi = t[-1]*1000

spec_db = 10*np.log10(spec)
filteredspec_db = 10*np.log10(filteredspec)

vmax = max(spec_db.max(), filteredspec_db.max())
vmin = vmax - 80.0  # Clip display of values below 80 dB

linecolor = 'k'
cmap = plt.cm.coolwarm

plt.figure(figsize=(4.0, 4.5))
plt.subplot(211)
plt.plot(time, pressure, color=linecolor, alpha=0.8)
plt.xlim(tlo, thi)
plt.ylim(0, 10)
plt.ylabel("Pressure (MPa)")
plt.grid(alpha=0.25)

plt.subplot(212)
plt.pcolormesh(1000*t, f/1000, spec_db, vmin=vmin, vmax=vmax,
               cmap=cmap, shading='gouraud')
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
plt.pcolormesh(1000*t, f/1000, filteredspec_db, vmin=vmin, vmax=vmax,
               cmap=cmap, shading='gouraud')
plt.axhline(cutoff/1000, color='k', linestyle='--', linewidth=1, alpha=0.5)
plt.xlim(tlo, thi)
plt.ylabel('Frequency (kHz)')
plt.xlabel('Time (ms)')
plt.tight_layout()
plt.savefig("pressure_example_filtered.pdf")
