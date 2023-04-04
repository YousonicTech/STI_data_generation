# -*- coding: utf-8 -*-
"""
@file      :  calCutTime.py
@Time      :  2022/9/6 18:42
@Software  :  PyCharm
@summary   :
@Author    :  Bajian Xiang
"""

import os
import librosa
import soundfile
import pickle
import glob

def cal_time(original):
    i = 0
    maxx = original.max()
    while original[i] < maxx / 1000:
        i += 1
    return i


if __name__ == "__main__":
    rir_dict = {}
    original_file_head = '/data2/wzd/0906_2000Hz_Dataset/2000_Hz_Online/'
    for file in os.listdir(original_file_head):
        path = os.path.join(original_file_head, file)
        # path = '/data2/wzd/0906_2000Hz_Dataset/2000_Hz_Online/ron-cooke-hub-university-york'
        wav_files = glob.glob(path + '/*.wav')
        new_path = path.replace(original_file_head, '/data2/xbj/rir_time_dict/0913_2000hzRIR/')
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        for rir_path in wav_files:
            # rir_path = './ron-cooke-hub-university-york/tstr_ir_4.wav'
            rir, rir_sr = librosa.load(rir_path, sr=16000, mono=False)
            time = cal_time(rir)
            rir_name = rir_path.split('/')[-1].split('.')[0] # tstr_ir_4
            rir_dict[rir_name] = time
            print('find --', rir_dict, ':', time)
    output = open('./rir_time.pkl', 'wb')
    pickle.dump(rir_dict, output)
    print('*** Final dcit ***')
    print(rir_dict)
