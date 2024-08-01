import numpy as np
from biosppy.signals import tools as tools

def normalize(signal):
    sigMax = signal.max()
    sigMin = signal.min()
    sigRan = sigMax-sigMin
    signal = signal*2/sigRan
    signal -= (signal.max() - 1)
    return signal
    
def normalize_meantf(signal):
    signal = normalize(signal)
    mean =  np.mean(signal)
    signal -= np.ones(len(signal))*mean
    return signal

def filter_ecg(signal, sampling_rate, frequency=[3,45]):
    
    signal = np.array(signal)
    order = int(0.3 * sampling_rate)
    filtered, _, _ = tools.filter_signal(signal=signal,
                                  ftype='FIR',
                                  band='bandpass',
                                  order=order,
                                  frequency=frequency,
                                  sampling_rate=sampling_rate)
    
    return filtered

def filter_ppg(signal, sampling_rate,frequency=[0.5,8]):
    
    signal = np.array(signal)
    sampling_rate = float(sampling_rate)
    filtered, _, _ = tools.filter_signal(signal=signal,
                                  ftype='butter',
                                  band='bandpass',
                                  order=4, #3
                                  frequency=frequency,
                                  sampling_rate=sampling_rate)

    return filtered

