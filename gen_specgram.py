import torch
import librosa
import acoustics
import numpy as np
import scipy
from acoustics.signal import bandpass, highpass

import matplotlib.pyplot as plt
from acoustics.bands import (_check_band_type, octave_low, octave_high, third_low, third_high)
import cv2

import torch
import torch.nn as nn
import torchaudio.functional as F
import torchaudio
from torchaudio.transforms import Spectrogram, InverseSpectrogram, Resample, AmplitudeToDB, TimeMasking
import matplotlib.pyplot as plt
from scipy.io.wavfile import write


def separate_complex(mm):
    # return torch.cat([mm.real, mm.imag], dim=-1)
    mm = mm.log10()
    real = mm.real.unsqueeze(0)
    imag = mm.imag.unsqueeze(0)
    mm = torch.cat([real, imag], dim=0)
    return mm


class PipeLine(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_fft = 512
        self.n_hop = 256
        self.freq = 16000
        self.spec = Spectrogram(n_fft=self.n_fft, hop_length=self.n_hop, power=None)
        self.inverseSpec = InverseSpectrogram(n_fft=self.n_fft, hop_length=self.n_hop)

    def forward(self, chunk, fs):
        #freq = 1000
        # 送进来的应该是一个4s的片段,类型为numpy array,先滤波,再downsample,最后画语谱图
        chunk = torch.from_numpy(chunk).squeeze()
        chunk = chunk.float()
        # 1. 滤波
        # chunk = bandpass(chunk, int(freq / 1.414), int(freq * 1.414), fs)
        # 2. downsample
        # chunk = F.resample(torch.from_numpy(chunk), orig_freq=fs, new_freq=int(freq * 1.414 * 2))
        if fs != self.freq:
            chunk = F.resample(torch.from_numpy(chunk), orig_freq=fs, new_freq=self.freq)
        # 3. 语谱图
        chunk = self.spec(chunk)
        # 4. 分割 spec
        chunk = separate_complex(chunk).squeeze() # [2, 257, 222]
        return chunk


class PipeLineNew(nn.Module):
    """新的数据集，500, 1k, 2k, 4k分别占一个维度,直接存torch.complex"""
    def __init__(self):
        super().__init__()
        self.n_fft = 512
        self.n_hop = 256
        self.freq = 16000
        self.spec = Spectrogram(n_fft=self.n_fft, hop_length=self.n_hop, power=None)
        self.inverseSpec = InverseSpectrogram(n_fft=self.n_fft, hop_length=self.n_hop)

    def forward(self, chunk, fs):
        # 送进来的应该是一个4s的片段,类型为numpy array,先滤波,再downsample,最后画语谱图
        chunk = torch.from_numpy(chunk).squeeze()
        if fs != self.freq:
            chunk = F.resample(torch.from_numpy(chunk), orig_freq=fs, new_freq=self.freq)
        chunk = chunk.float()
        save_tensor = []
        for freq, fft in zip([500, 1000, 2000, 4000], [256, 512, 1024, 2048]):
            # 1. 滤波
            temp_chunk = bandpass(chunk, int(freq / 1.414), int(freq * 1.414), fs)
            # 2. downsample
            temp_chunk = F.resample(torch.from_numpy(temp_chunk), orig_freq=fs, new_freq=int(freq * 1.414 * 2))
            # 3. 语谱图
            spec = Spectrogram(n_fft=fft, hop_length=fft//2, power=None)
            temp_chunk = spec(temp_chunk).squeeze()
            save_tensor.append(temp_chunk)
        return save_tensor


def paint_single(spec, fs, time):
    plt.figure(figsize=(20, 10), dpi=150)
    plt.imshow(spec, origin='lower')

    max_freq, min_freq = fs[-1], fs[0]
    max_t, min_t = time[-1], time[0]

    margin = 10
    x_margin = 4

    freq_temp = [int((max_freq - min_freq) / margin * i + min_freq) for i in range(margin)]
    freq_temp.append(max_freq)
    y_ticks_temp = [int(len(fs) / margin * x) for x in range(margin)]
    y_ticks_temp.append((len(fs)))

    t_temp = [int((max_t - min_t) / x_margin * i + min_t) for i in range(x_margin)]
    t_temp.append(round(max_t))
    x_ticks_temp = [int(len(time) / x_margin * x) for x in range(x_margin)]
    x_ticks_temp.append(len(time))

    plt.yticks(y_ticks_temp, freq_temp)
    plt.xticks(x_ticks_temp, t_temp)

    plt.ylabel('Freq / Hz')
    plt.xlabel('Time / s')
    plt.show()


def paint(spectrograms, i, fs_each_band, time):
    plt.figure(figsize=(20, 10), dpi=150)

    # plt.figure(dpi=150)
    plt.imshow(spectrograms[i], origin='lower')
    if i == 0:
        plt.xticks(fontsize=6, rotation=45)
    margin = 10
    x_margin = 4
    max_freq, min_freq = fs_each_band[i][-1], fs_each_band[i][0]
    max_t, min_t = time[i][-1], time[i][0]
    # print(min_freq, max_freq)
    #
    freq_temp = [int((max_freq - min_freq) / margin * i + min_freq) for i in range(margin)]
    freq_temp.append(np.array(int(max_freq)))
    y_ticks_temp = [int(len(fs_each_band[i]) / margin * x) for x in range(margin)]
    y_ticks_temp.append(np.array(len(fs_each_band[i])))
    #
    t_temp = [int((max_t - min_t) / x_margin * i + min_t) for i in range(x_margin)]
    t_temp.append(np.array(round(max_t)))
    x_ticks_temp = [int(len(time[i]) / x_margin * x) for x in range(x_margin)]
    x_ticks_temp.append(np.array(len(time[i])))
    plt.yticks(y_ticks_temp, freq_temp)
    plt.xticks(x_ticks_temp, t_temp)
    plt.savefig('./save_fig/highpass_final_%d_band.jpg' % (i), bbox_inches='tight', pad_inches=0.0)
    # freq_tick = [fs_each_band[i][c*10] for c in range(25)]
    # plt.yticks(freq_tick)
    # plt.xticks(x_ticks_temp)

    # plt.show()


def Filter_Downsample_Spec(waveform, fs):
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    channel = len(waveform)
    nframes = len(waveform[0])
    raw_signal = waveform.T
    bands = acoustics.signal.OctaveBand(fstart=125, fstop=4000, fraction=1).nominal

    band_type = _check_band_type(bands)
    # print(band_type, end=', ')

    if band_type == 'octave':
        low = octave_low(bands[0], bands[-1])
        high = octave_high(bands[0], bands[-1])
    elif band_type == 'third':
        low = third_low(bands[0], bands[-1])
        high = third_high(bands[0], bands[-1])

    # for nch in range(channel):
    nch = 0
    filtered_signal = np.zeros((bands.size, nframes))
    for band in range(bands.size):
        # 信号，频率下限，频率上限， 采样率
        print("low:", low[band], "high:", high[band])
        filtered_signal[band] = bandpass(raw_signal[:, nch], low[band], high[band], fs, order=bands.size)
        filtered_signal[band] = highpass(filtered_signal[band], low[band], fs, order=6)
        plt.figure()
        plt.plot(filtered_signal[band])
        plt.clf()

    downsample_signal = []
    for i in range(len(bands)):
        temp_rate = 2 * high[i]
        temp_data = filtered_signal[i]
        number_of_samples = round(len(temp_data) * float(temp_rate) / fs)
        downsample_signal.append(scipy.signal.resample(temp_data, number_of_samples))
        plt.figure()
        plt.plot(downsample_signal[i])
        plt.clf()

    spectrograms = []
    nfft = 256
    fs_each_band = []
    time = []

    for i in range(len(bands)):
        spec, freq, t, _ = plt.specgram(downsample_signal[i], NFFT=nfft, Fs=high[i] * 2, window=np.hanning(M=nfft),
                                        scale_by_freq=True)
        print(spec.shape)
        plt.clf()
        spec = 10 * np.log10(spec)
        high_index = round((high[i] - low[i]) / (freq[1] - freq[0])) - 1
        spectrograms.append(spec)
        fs_each_band.append(freq)
        time.append(t)

    return spectrograms, fs_each_band, time


def All_Frequency_Spec(waveform, fs):
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    raw_signal = waveform[0]
    nfft = 512

    spec, freq, t, _ = plt.specgram(raw_signal, NFFT=nfft, Fs=fs, window=np.hanning(M=nfft), scale_by_freq=True)
    print(spec.shape)
    plt.clf()
    spec = 10 * np.log10(spec)

    return spec, freq, t


if __name__ == "__main__":
    # path="/Users/bajianxiang/Desktop/internship/koli-national-park-winter_koli_snow_site4_1way_mono_1_koli-national-park-winter_粤语女声_5_TIMIT_b011_30_40_20dB.wav"
    # data, fs = librosa.load(path, sr=None, mono=False)
    # # spectrograms, fs_each_band, time = Filter_Downsample_Spec(data, fs)
    # spectrograms,fs_each_band,time = All_Frequency_Spec(data,fs)
    # # paint_single(spectrograms, fs_each_band, time)
    # # print(len(spectrograms))

    audio_path = "/Users/bajianxiang/Desktop/internship/code_split_wav/koli-national-park" \
                 "-winter_koli_snow_site4_1way_bformat_1_koli-national-park-winter_东北话男声_1_TIMIT_a017_110_120_0dB.wav"
    pip = PipeLineNew()
    y, fs = librosa.load(audio_path, sr=None, mono=False)
    reverb = pip(y, fs)
    print(reverb.shape)
