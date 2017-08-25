import numpy as np
from scipy.signal import firwin2, firls, freqz, kaiser_atten, kaiser_beta, remez
import matplotlib.pyplot as plt


fs = 2000
freqs = [0, 45,  59,  61, 75, 150, 175, 1000]
gains = [1,  1, 0.1, 0.1,  1,   1,   0,    0]

numtaps = 175

taps_none = firwin2(numtaps, freqs, gains, nyq=0.5*fs, window=None)
taps_fw = firwin2(numtaps, freqs, gains, nyq=0.5*fs)

atten = kaiser_atten(numtaps, 20/(0.5*fs))
beta = kaiser_beta(atten)
taps_k = firwin2(numtaps, freqs, gains, nyq=0.5*fs, window=('kaiser', beta))

#bands = [0, 5, 5, 25, 25, 45, 45,   59,   59,   61,   61, 75, 75, 150, 150, 175, 175, 1000]
#des   = [0, 0, 0,  1,  1,  1,  1,  0.1,  0.1,  0.1,  0.1,  1,  1,   1,   1,   0,   0,    0]
bands = [0, 45,   59,   61, 75, 150, 175, 1000]
des   = [1,  1,  0.1,  0.1,  1,   1,   0,    0]
taps_ls = firls(numtaps, bands, des, nyq=0.5*fs)

rbands = [0, 45, 59, 61, 75, 150, 175, 1000]
rdes   = [    1,    0.1,       1,         0]
rwt    = [    3,      1,       3,         1]
taps_r = remez(numtaps, rbands, rdes, Hz=fs)
#taps_rw = remez(numtaps, rbands, rdes, weight=rwt, Hz=fs)

w_none, h_none = freqz(taps_none, 1, worN=2000)
w_fw, h_fw = freqz(taps_fw, 1, worN=2000)
w_k, h_k = freqz(taps_k, 1, worN=2000)

w_ls, h_ls = freqz(taps_ls, 1, worN=2000)

w_r, h_r = freqz(taps_r, 1, worN=2000)
#w_rw, h_rw = freqz(taps_rw, 1, worN=2000)

plt.figure(figsize=(8.0, 3.5))
plt.plot(freqs, gains, 'k--', alpha=0.5, linewidth=1, label='ideal (firwin2)')

for xlim, y in zip(np.reshape(rbands, (-1, 2)), rdes):
    if xlim[0] == 0:
        label="ideal (firls and remez)"
    else:
        label=None
    #plt.plot(xlim, [y,y], 'k', alpha=0.25, linewidth=5, label=label)

plt.plot(0.5*fs*w_fw/np.pi, np.abs(h_fw), alpha=0.9, label='Hamming window')
plt.plot(0.5*fs*w_k/np.pi, np.abs(h_k), alpha=0.6, label=r'Kaiser window, $\beta$=%.2f' % beta)
plt.plot(0.5*fs*w_none/np.pi, np.abs(h_none), alpha=0.9, label='no window')
#plt.plot(0.5*fs*w_ls/np.pi, np.abs(h_ls), alpha=0.6, label='firls')
#plt.plot(0.5*fs*w_r/np.pi, np.abs(h_r), 'k', alpha=0.5, label='remez')
#plt.plot(0.5*fs*w_rw/np.pi, np.abs(h_rw), 'k', alpha=0.5, dashes=(6, 0.5), label='remez, with weights %s' % (rwt,))
plt.xlim(0, 210)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.legend(framealpha=1, shadow=True, loc=(0.35, 0.05))
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()
