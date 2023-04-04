# -*- coding: utf-8 -*-
"""
@file      :  0929_addNoiseNSplitWav.py
@Time      :  2022/9/29 09:45
@Software  :  PyCharm
@summary   :  1. Load the waves WITHOUT noise, generate pt file for it using ORIGINAL GT extracted from csv file.
              2. Add 0/6/20dB noise to the waves and Generate the estimated STI as ground truth.
@Author    :  Bajian Xiang
"""
import random

import librosa
import numpy as np
from scipy.signal import lfilter
import gtg
import warnings
import splweighting
import wave
import glob
import os
import re
import math
import torch
import pandas as pd
import argparse
from gen_specgram import All_Frequency_Spec
from cutTimeDict import rir_dict
from calSTIwithNOISE import sti
import soundfile as sf

warnings.filterwarnings('ignore')

# /mnt/sda/xbj/code_split_wav/0929_slice_pt_with_noise
# /data/xbj/0929_STI_catTIMIT_withNOISE_DATA/train/Four_Config
parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('--csv_file', type=str,
                    default="/data/data1/0923_STI_Dataset/0923_STI/Dev/0923_STI.csv")
parser.add_argument('--dir_str', type=str,
                    default="/data/data1/0923_STI_Dataset/0923_STI/Dev/Speech/Eight_Config/")
parser.add_argument('--save_dir', type=str,
                    default="/data/xbj/0929_STI_catTIMIT_withNOISE_DATA/train/Eight_Config/")
parser.add_argument('--rir_path', type=str, default="/data/xbj/STI_RIR/Rir_dataset/")
parser.add_argument('--clean_speech_path', type=str, default="/data/xbj/new_timit/catTIMIT/")
parser.add_argument('--noise_txt', type=str, default="/data/xbj/STI_Noise/all_noise_all.txt")
parser.add_argument('--save_speech_dir', type=str, default="/data2/xbj/STI_Noise_Wav/")
parser.add_argument('--chunk_length', type=int, default=4)
parser.add_argument('--chunk_overlap', type=float, default=0.5)


def SPLCal(x):
    Leng = len(x)
    pa = np.sqrt(np.sum(np.power(x, 2)) / Leng)
    p0 = 2e-5
    spl = 20 * np.log10(pa / p0)
    return spl


def get_rir_name(wav_file_name):
    # Four_Config_3000CStreetGarageStairwell_leftFour_Config__DR4_FCAG0_SX63_TIMIT_S_1000dB.wav
    room_name = wav_file_name.split('/')[-1].split('.')[0]
    config_name = wav_file_name.split('/')[-2]
    rir_name = room_name.split(config_name)[1].strip('_')
    speech_name = wav_file_name.split(config_name)[-1].split('TIMIT')[0].strip('_') + '.wav'
    return rir_name, speech_name


def cut_begin(wave, cut_time):
    res = wave[cut_time:]
    return res


def get_rir_array(rir_path, file_name):
    room_name = file_name.split('/')[-1].split('.')[0]
    config_name = file_name.split('/')[-2]
    rir_name = room_name.split(config_name)[1].strip('_')

    rir_file_path = os.path.join(rir_path, config_name, rir_name + '.wav')

    if not os.path.exists(rir_file_path):
        raise OSError("Cannot find:", rir_file_path)

    rir, fs = librosa.load(rir_file_path, sr=None, mono=False)

    if fs != 16000:
        rir = librosa.resample(rir, fs, 16000)

    return rir


def v_addnoise(Voice, noise, snr):
    # 直接从gen_corpus_dataset里拿出来的
    Data = Voice
    data = noise
    data = np.tile(data, 10)
    # if fs == Fs and len(Data)<=len(data):
    Average_Energy = np.sum(Data ** 2) / len(Data)
    average_energy = np.sum(data ** 2) / len(data)
    k = math.sqrt(Average_Energy / average_energy / 10 ** (snr * 0.1))
    num = random.randint(16000, len(data) - len(Data) - 16000)
    Data_new = Data + data[num:len(Data) + num] * k
    return Data_new


people_choose = ['DR8_MMPM0', 'DR5_FSDC0', 'DR7_MTAB0', 'DR5_MJFH0', 'DR7_MVRW0', 'DR7_MDKS0', 'DR7_MMWS1', 'DR4_FDKN0',
                 'DR5_FBMH0', 'DR7_FKDE0', 'DR5_FSMM0', 'DR7_FPAC0', 'DR4_MPEB0', 'DR5_MJDM0', 'DR4_MJXL0', 'DR4_MFRM0',
                 'DR6_MCAE0', 'DR7_MTKD0', 'DR5_MBGT0', 'DR6_MESJ0', 'DR7_FLET0', 'DR4_MBMA0', 'DR7_MCLK0', 'DR4_MGJC0',
                 'DR7_MWRP0', 'DR7_MWRE0', 'DR8_FCLT0', 'DR5_FGDP0', 'DR6_MDRD0', 'DR8_MMLM0', 'DR4_MJLB0', 'DR4_FPAF0',
                 'DR5_FMPG0', 'DR4_MJDC0', 'DR7_MTMN0', 'DR6_MEAL0', 'DR5_MSEM1', 'DR7_MMDG0', 'DR5_MJWG0', 'DR5_MJXA0',
                 'DR8_FMBG0', 'DR6_MPGR1', 'DR4_MJJJ0', 'DR7_MJFR0', 'DR8_FCEG0', 'DR7_MGAR0', 'DR7_MJAI0', 'DR7_MREM0',
                 'DR7_MSAH1', 'DR5_MREW1', 'DR7_FJRP1', 'DR7_FJHK0', 'DR5_FDTD0', 'DR6_FBCH0', 'DR7_MBAR0', 'DR7_MDPB0',
                 'DR7_MDED0', 'DR7_MDLM0', 'DR5_MGSH0', 'DR4_MGAG0', 'DR5_MDSJ0', 'DR4_MMGC0', 'DR8_MKRG0', 'DR6_FTAJ0',
                 'DR5_MMVP0', 'DR6_FPAD0', 'DR4_MTRC0', 'DR4_FSSB0', 'DR4_MNET0', 'DR4_MMBS0', 'DR4_FJWB1', 'DR5_FBJL0',
                 'DR5_FDMY0', 'DR6_FKLC1', 'DR4_MRAB1', 'DR7_MGAK0', 'DR8_MEJS0', 'DR5_FSJG0', 'DR4_MJWS0', 'DR5_MHMG0',
                 'DR5_MEGJ0', 'DR8_MBCG0', 'DR7_MHXL0', 'DR5_MEWM0', 'DR8_MBSB0', 'DR7_FLEH0', 'DR8_MRLK0', 'DR5_FEXM0',
                 'DR4_MJEE0', 'DR5_MRAV0']


def is_file_people_choose(file_name):
    for name in people_choose:
        if name in file_name:
            return True
    return False


class Totensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, clean, ddr, t60, meanT60 = sample['image'], sample['clean'], sample['ddr'], sample['t60'], sample[
            'MeanT60']

        # image, ddr, t60 = sample['image'], sample['ddr'], sample['t60']
        image = torch.from_numpy(image)
        clean = torch.from_numpy(clean)
        ddr = ddr.astype(float)
        t60 = t60.astype(float)
        meanT60 = meanT60.astype(float)
        # image = image.transpose((2, 0, 1))
        return {'image': image,
                'clean': clean,
                'ddr': torch.from_numpy(ddr),
                't60': torch.from_numpy(t60),
                "MeanT60": torch.from_numpy(meanT60)
                }


if __name__ == "__main__":

    args = parser.parse_args()

    save_dir = args.save_dir
    dir_str = args.dir_str
    csv_file = args.csv_file
    chunk_length = args.chunk_length
    chunk_overlap = args.chunk_overlap
    rir_path = args.rir_path
    clean_speech_path = args.clean_speech_path
    csv_data = pd.read_csv(csv_file)
    noise_txt = args.noise_txt
    save_speech_dir = args.save_speech_dir
    print('csv data:', csv_data)

    with open(noise_txt, 'r') as f:
        noise_list = [lines.split('\n')[0] for lines in f.readlines()]
        f.close()

    if not os.path.exists(args.save_dir):
        os.makedirs(save_dir)

    for file_name in glob.glob(dir_str + r"/*.wav"):
        """        
        E.g., file_name =  "/data/STI_New_Wav/Dev/Speech/Four_Config/
        Four_Config_3000CStreetGarageStairwell_left_Four_Config_DR4_FCAG0_SX63_TIMIT_S_1000dB.wav"
        """
        if not is_file_people_choose(file_name):
            continue
        # We just need 90 speech in this, so pick 90 out.

        rir_array = get_rir_array(rir_path, file_name)

        f = wave.open(file_name, "rb")
        params = f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        print('file_name:', file_name)

        rir_name, speech_name = get_rir_name(file_name)
        print('info:', nchannels, sampwidth, framerate, nframes / framerate)

        if rir_name in rir_dict.keys():
            cut_index = rir_dict[rir_name]
        else:
            print('DO NOT FIND:', rir_name, '-- for -- ', file_name)
            break

        str_data = f.readframes(nframes)
        f.close()
        wave_data = np.frombuffer(str_data, dtype=np.int16)
        if len(wave_data) <= 40000:
            continue
        # TODO 切掉RIR前面的部分
        wave_data = cut_begin(wave_data, cut_index)

        wave_data.shape = -1, nchannels
        wave_data = wave_data.T
        audio_time = nframes / framerate
        chan_num = 0
        count = 0

        # TODO 读取干净语音，并进行相同的操作
        temp_clean_path = clean_speech_path + speech_name  # /data/xbj/TIMIT/DR4_FDKN0_SI1081.wav
        if not os.path.exists(temp_clean_path):
            print('not find clean:', temp_clean_path)
            break
        # 读取clean_wav的data
        else:
            f = wave.open(temp_clean_path, "rb")
            params = f.getparams()
            cnchannels, csampwidth, cframerate, cnframes = params[:4]

            clean_str_data = f.readframes(cnframes)
            f.close()
            clean_wave_data = np.frombuffer(clean_str_data, dtype=np.int16)  # ndarray(500228)

            clean_wave_data.shape = -1, cnchannels
            clean_wave_data = clean_wave_data.T  # (1, 500228)
            clean_audio_time = cnframes / cframerate
            clean_chan_num = 0
            clean_count = 0
        # Four_Config_3000CStreetGarageStairwell_leftFour_Config__DR4_FCAG0_SX63_TIMIT_S_1000dB.wav
        new_file_name = (file_name.split("\\")[-1]).split(".")[0]
        new_file_name = new_file_name.split("/")[-1]
        ## process each channel of audio

        for audio_samples_np, clean_audio_samples_np in zip(wave_data, clean_wave_data):

            whole_audio_SPL = SPLCal(audio_samples_np)  # 应该是算这个片段中语音出现的长度

            available_part_num = (audio_time - chunk_overlap) // (
                    chunk_length - chunk_overlap)  # 4*x - (x-1)*0.5 <= audio_time    x为available_part_num

            if available_part_num == 1:
                cut_parameters = [chunk_length]
            else:
                cut_parameters = np.arange(chunk_length,
                                           (chunk_length - chunk_overlap) * available_part_num + chunk_overlap,
                                           chunk_length)  # np.arange()函数第一个参数为起点，第二个参数为终点，第三个参数为步长（10秒）

            start_time = int(0)  # 开始时间设为0
            count = 0
            # 开始存储pt文件
            dict = {}
            save_data = []
            for t in cut_parameters:
                stop_time = int(t)  # pydub以毫秒为单位工作
                start = int(start_time * framerate)
                end = int((start_time + chunk_length) * framerate)

                audio_chunk = audio_samples_np[start:end]
                clean_audio_chunk = clean_audio_samples_np[start:end]

                chunk_spl = SPLCal(audio_chunk)
                if whole_audio_SPL - chunk_spl >= 20:
                    continue

                count += 1

                chunk_a_weighting = splweighting.weight_signal(audio_chunk, framerate)
                clean_chunk_a_weighting = splweighting.weight_signal(clean_audio_chunk, cframerate)

                chunk_result, _, _ = All_Frequency_Spec(chunk_a_weighting, framerate)
                clean_chunk_result, _, _ = All_Frequency_Spec(clean_chunk_a_weighting, cframerate)

                chan = chan_num + 1

                config = new_file_name.split("_")[0] + "_" + new_file_name.split("_")[1]
                if config == "dirac":
                    config = new_file_name.split("_")[0]  # +"_" + new_file_name.split("_")[1]
                    room = new_file_name.split(config)[1][1:-1]
                else:
                    config = new_file_name.split("_")[0] + "_" + new_file_name.split("_")[1]
                    room = new_file_name.split(config)[1].strip('_')
                print('new_file_name:', new_file_name)

                a = (csv_data['Room:'] == room).values
                b = (csv_data['Room Config:'] == config).values

                data = csv_data[a]
                T60_data = data.loc[:, ['T60:']]
                FB_T60_data = data.loc[:, ['FB T60:']]
                FB_T60_M_data = data.loc[:, ['FB T60 Mean (Ch):']]
                DDR_each_band = np.array([0 for i in range(30)])
                T60_each_band = (T60_data.values).reshape(-1)
                MeanT60_each_band = np.array([FB_T60_data, FB_T60_M_data])
                image = chunk_result
                clean_image = clean_chunk_result
                print('-- Reverb Image: ', image.shape, ' -- Clean Image:', clean_image.shape, 'sti:',
                      T60_each_band[10])
                sample = {'image': image, 'clean': clean_image, 'ddr': DDR_each_band, 't60': T60_each_band,
                          "MeanT60": MeanT60_each_band}
                transform = Totensor()
                sample = transform(sample)

                save_data.append(sample)

                start_time = start_time + chunk_length - chunk_overlap  # 开始时间变为结束时间前1s---------也就是叠加上一段音频末尾的4s

            if len(save_data) != 0:
                pt_file_name = os.path.join(save_dir, new_file_name + '-' + str(chan_num) + '.pt')
                dict[new_file_name + '-' + str(chan_num)] = save_data
                torch.save(dict, pt_file_name)
            chan_num = chan_num + 1

        print('----------------finish Loading original wav----------------')

        # 添加噪声，生成噪声对应的PT

        # noise_list = ['/data/xbj/STI_Noise/Noise/babble_noise.wav', ...]

        wave_data, fs_new = librosa.load(file_name, sr=None, mono=False)

        temp_noise_list = random.sample(noise_list, 20)  # 随机选20种噪声
        for temp_noise_path in temp_noise_list:
            print('** Add Noise type:', temp_noise_path, end='  | ')
            temp_noise, noise_fs = librosa.load(temp_noise_path, sr=None, mono=False)
            if len(temp_noise) == 2:
                # take one channel only
                temp_noise = temp_noise[0]
            if noise_fs != 16000:
                temp_noise = librosa.resample(temp_noise, noise_fs, 16000)

            for snr in [0, 6, 20]:
                try:
                    speech_with_noise = v_addnoise(wave_data, temp_noise, snr)
                except TypeError as e:
                    print("GET wave_data and temp_noise with shape:", wave_data.shape, temp_noise.shape)

                estimated_sti = sti(rir_array, 16000, wave_data, speech_with_noise)

                print('SNR = ', snr, 'sti_estimate: ', estimated_sti)
                whole_audio_SPL = SPLCal(speech_with_noise)

                available_part_num = (audio_time - chunk_overlap) // (
                        chunk_length - chunk_overlap)  # 4*x - (x-1)*0.5 <= audio_time    x为available_part_num

                if available_part_num == 1:
                    cut_parameters = [chunk_length]
                else:
                    cut_parameters = np.arange(chunk_length,
                                               (chunk_length - chunk_overlap) * available_part_num + chunk_overlap,
                                               chunk_length)  # np.arange()函数第一个参数为起点，第二个参数为终点，第三个参数为步长（10秒）

                start_time = int(0)  # 开始时间设为0
                count = 0
                # 开始存储pt文件
                dict = {}
                save_data = []

                for t in cut_parameters:
                    stop_time = int(t)  # pydub以毫秒为单位工作
                    start = int(start_time * framerate)
                    end = int((start_time + chunk_length) * framerate)

                    audio_chunk = speech_with_noise[start:end]  # 音频切割按开始时间到结束时间切割
                    clean_audio_chunk = clean_wave_data[0][start:end]

                    ##ingore chunks with no audio
                    chunk_spl = SPLCal(audio_chunk)
                    if whole_audio_SPL - chunk_spl >= 20:
                        continue

                    count += 1

                    chunk_a_weighting = splweighting.weight_signal(audio_chunk, framerate)
                    clean_chunk_a_weighting = splweighting.weight_signal(clean_audio_chunk, cframerate)

                    ##gammatone
                    chunk_result, _, _ = All_Frequency_Spec(chunk_a_weighting, framerate)
                    clean_chunk_result, _, _ = All_Frequency_Spec(clean_chunk_a_weighting, cframerate)

                    chan = chan_num + 1

                    image = chunk_result
                    clean_image = clean_chunk_result
                    T60_each_band[10] = estimated_sti

                    sample = {'image': image, 'clean': clean_image, 'ddr': DDR_each_band, 't60': T60_each_band,
                              "MeanT60": MeanT60_each_band}
                    transform = Totensor()
                    sample = transform(sample)

                    save_data.append(sample)

                    start_time = start_time + chunk_length - chunk_overlap  # 开始时间变为结束时间前1s---------也就是叠加上一段音频末尾的4s

                print('-- Reverb Image: ', image.shape, ' -- Clean Image:', clean_image.shape, 'sti:',
                      T60_each_band[10])
                noise_name = temp_noise_path.split('/')[-1].split('.wav')[0]
                if len(save_data) != 0:
                    pt_file_name = os.path.join(save_dir,
                                                new_file_name + '-' + str(chan_num) + '-' + noise_name + '-' + str(
                                                    snr) + 'dB.pt')

                    wav_config = 'STI-' + str(int(estimated_sti * 10))  # STI-1, STI-2, STI-3...
                    str_sti = str(estimated_sti)[2:6]  # 4085
                    wave_config_dir = os.path.join(save_speech_dir, wav_config)
                    if not os.path.exists(wave_config_dir):
                        os.mkdir(wave_config_dir)
                    wav_file_name = os.path.join(wave_config_dir,
                                                 new_file_name + '-' + str(chan_num) + '-' + noise_name + '-' + str(
                                                     snr) + 'dB-' + str_sti + '.wav')
                    sf.write(wav_file_name, speech_with_noise, 16000)
                    dict[new_file_name + '-' + str(chan_num)] = save_data
                    torch.save(dict, pt_file_name)
                chan_num = chan_num + 1

            print('----------------finish Loading noise wav----------------')
