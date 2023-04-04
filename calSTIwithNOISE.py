# -*- coding: utf-8 -*-
"""
@file      :  STI_with_noise.py
@Time      :  2022/9/28 16:33
@Software  :  PyCharm
@summary   :
@Author    :  Bajian Xiang
"""


import acoustics
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)
from acoustics.signal import bandpass,highpass

from scipy.interpolate import interp1d

import numpy as np
import torch


def SNR_singlech(clean_sig, with_noise_sig):
    estimate_noise = with_noise_sig - clean_sig
    SNR = 10 * np.log10(np.sum(clean_sig ** 2) / np.sum(estimate_noise ** 2))

    return SNR


def cal_SNR_for_each_freq(clean, signal, fs=16000):
    """计算每个倍频带的信噪比.

    Parameters
    ----------
    clean : numpy.array
            干净的人声,
    signal: numpy.array
            含噪人声,
    fs: int
            采样率.
    """
    # 都先消除直流分量和归一化
    signal = signal - np.mean(signal)  # 消除直流分量
    signal = signal / np.max(np.abs(signal))  # 幅值归一化
    clean = clean - np.mean(clean)
    clean = clean / np.max(np.abs(clean))

    bands = acoustics.signal.OctaveBand(fstart=125, fstop=8000, fraction=1).nominal
    # bands = array([ 125.,  250.,  500., 1000., 2000., 4000., 8000.]
    band_type = _check_band_type(bands)

    if band_type == 'octave':
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
    elif band_type == 'third':
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])

    clean_filtered = np.zeros((bands.size, len(signal)))
    signal_filtered = np.zeros((bands.size, len(signal)))

    # 分别对clean和signal进行分频段滤波
    for band in range(bands.size):
        # 信号，频率下限，频率上限， 采样率
        # band from 0 to 6, totally 7
        if high[band] <= fs / 2:
            # print('bandpass:', low[band], '-' ,high[band])
            clean_filtered[band] = bandpass(clean, low[band], high[band], fs, order=6)
            signal_filtered[band] = bandpass(signal, low[band], high[band], fs, order=6)
        else:
            # print('highpass:', low[band])
            clean_filtered[band] = highpass(clean, low[band], fs, order=6)

    # 对滤波后的信号进行信噪比计算
    snr_res = []
    for freq in range(clean_filtered.shape[0]):
        temp_snr = SNR_singlech(clean_filtered[freq], signal_filtered[freq])
        snr_res.append(temp_snr)
    return snr_res


def pow2db(x):
    return 10 * np.log10(x)


def get_mtf(ir):
    return np.abs(np.fft.rfft(ir ** 2) / np.sum(ir ** 2))


def cal_sti(mti):
    alpha = torch.tensor((0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173))
    beta = torch.tensor((0.085, 0.078, 0.065, 0.011, 0.047, 0.095))
    mti = torch.from_numpy(mti)
    return torch.sum(alpha * mti) - torch.sum(beta * torch.sqrt(mti[1:] * mti[:-1]))


def modify_m(m, snr):
    # m.shape = [14, 7], need to be transposed first
    m = m.T
    for freq, i in enumerate(snr):
        m[freq] = m[freq] / (1 + 10 ** (-i / 10))
    return m.T


def sti(ir, fs, clean, signal):
    # 1. filter ir through octave bands
    # 2. get MTF through
    Nfc = 7  # 7 octaves from 125 - 8000Hz

    # MTF function.
    mtf = get_mtf(ir)

    modulation_freqs = [0.63, 0.80, 1.00, 1.25, 1.60, 2.00, 2.50, 3.15,
                        4.00, 5.00, 6.30, 8.00, 10.00, 12.50]

    freqs = np.linspace(0, fs // 2, mtf.shape[0])
    # freqs[-1] = 0 # No nyquist frequency

    m = np.zeros((len(modulation_freqs), Nfc))

    snr_res = cal_SNR_for_each_freq(clean=clean, signal=signal)

    for i in range(Nfc):
        # Old x is freqs, y is mtf[i, :], newx = modulation_freqs
        # m(i,:) = interp1(freqs,MTF_octband(1:end/2,i),modulation_freqs);
        interp = interp1d(freqs, mtf[:])
        m[:, i] = interp(modulation_freqs)

    m = modify_m(m, snr_res)
    # Convert each of the 98m values into an apparent SNR in dB
    SNR_apparent = pow2db(m / (1 - m))

    # Limit the range
    SNR_apparent_clipped = np.clip(SNR_apparent, -15, 15)

    # 增加: NORMALIZATION
    TI = (SNR_apparent_clipped + 15) / 30

    # 修改: 计算 MTI
    MTI = np.mean(TI, axis=0)

    # 修改: 计算STI
    sti_val = float(cal_sti(MTI))

    return sti_val