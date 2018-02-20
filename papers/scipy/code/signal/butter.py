from __future__ import division, print_function

from scipy.signal import butter, sosfilt, sosfiltfilt


def butter_lowpass(cutoff, fs, order):
    normal_cutoff = cutoff / (0.5*fs)
    sos = butter(order, normal_cutoff,
                 btype='low', output='sos')
    return sos


def butter_lowpass_filtfilt(data, cutoff, fs,
                            order):
    sos = butter_lowpass(cutoff, fs, order)
    y = sosfiltfilt(sos, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band',
                 output='sos')
    return sos


def butter_bandpass_filt(data, lowcut, highcut, fs,
                         order):
    sos = butter_bandpass(lowcut, highcut, fs, order)
    y = sosfilt(sos, data)
    return y
