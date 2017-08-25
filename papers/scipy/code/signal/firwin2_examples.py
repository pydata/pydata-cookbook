import numpy as np
from scipy.signal import firwin2, firls, freqz, kaiser_atten, kaiser_beta, remez
import matplotlib.pyplot as plt


fs = 2000
freqs = [0, 48,  59,  61, 72, 150, 175, 1000]
gains = [1,  1, 0.1, 0.1,  1,   1,   0,    0]

numtaps = 185

taps_none = firwin2(numtaps, freqs, gains, nyq=0.5*fs, window=None)
taps_fw = firwin2(numtaps, freqs, gains, nyq=0.5*fs)

atten = kaiser_atten(numtaps, 20/(0.5*fs))
beta = kaiser_beta(atten)
taps_k = firwin2(numtaps, freqs, gains, nyq=0.5*fs, window=('kaiser', beta))

bands = [0, 48,   59,   61, 72, 150, 175, 1000]
des   = [1,  1,  0.1,  0.1,  1,   1,   0,    0]
taps_ls = firls(numtaps, bands, des, nyq=0.5*fs)

rbands = [0, 48, 59, 61, 72, 150, 175, 1000]
rdes   = [    1,    0.1,       1,         0]
rwt    = [    3,      1,       3,         1]
taps_r = remez(numtaps, rbands, rdes, Hz=fs)
#taps_rw = remez(numtaps, rbands, rdes, weight=rwt, Hz=fs)

w_none, h_none = freqz(taps_none, 1, worN=2000)
w_fw, h_fw = freqz(taps_fw, 1, worN=2000)
w_k, h_k = freqz(taps_k, 1, worN=2000)

plt.figure(figsize=(4.0, 3.5))
plt.plot(freqs, gains, 'k--', alpha=0.5, linewidth=1, label='ideal')

plt.plot(0.5*fs*w_fw/np.pi, np.abs(h_fw), label='Hamming')
plt.plot(0.5*fs*w_k/np.pi, np.abs(h_k), label='Kaiser, $\\beta$=%.2f' % beta)
plt.plot(0.5*fs*w_none/np.pi, np.abs(h_none), label='no window')
plt.xlim(0, 210)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.legend(framealpha=1, shadow=True, loc='center right')
plt.grid(alpha=0.25)

plt.text(9, .08, "%d taps" % numtaps, fontsize=9)

plt.title('Filters designed with the window method', fontsize=10)
plt.tight_layout()

plt.savefig("firwin2_examples.pdf", dpi=600)

plt.show()
