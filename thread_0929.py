# -*- coding: utf-8 -*-
"""
@file      :  thread_0929.py
@Time      :  2022/9/29 16:54
@Software  :  PyCharm
@summary   :
@Author    :  Bajian Xiang
"""

# nohup python 0922_thread_STI_slicept.py  >> /mnt/sda/xbj/thread_0831_gen_data.log 2>&1 &

import datetime
import os
import threading
import glob


def execCmd(cmd):
    try:
        print("COMMAND -- %s -- BEGINS -- %s -- " % (cmd, datetime.datetime.now()))
        os.system(cmd)
        print("COMMAND -- %s -- ENDS -- %s -- " % (cmd, datetime.datetime.now()))
    except:
        print("Failed -- %s -- " % cmd)


# 如果只是路径变了的话，就改这3个地方,
# Don't forget the last '/' in those paths!!!!
# Carefully check!!!

dir_str_head = "/data/data1/0923_STI_Dataset/0923_STI/Dev/Speech/"
save_dir_head = "/data/xbj/0929_STI_catTIMIT_withNOISE_DATA/train/"

exist_files = os.listdir(save_dir_head)

dir_str = [os.path.join(dir_str_head, x) for x in os.listdir(dir_str_head) if x not in exist_files]

save_dir = [os.path.join(save_dir_head, x) for x in os.listdir(dir_str_head) if x not in exist_files]

def get_csv_dir(dir_str):
    csv_dir = []
    for i in dir_str:
        temp_csv = glob.glob(i+'/*.csv')
        if len(temp_csv) == 1:
            csv_dir.extend(temp_csv)
        else:
            a, b = temp_csv
            a = a.split('/')[-1].split('_')[0].split('T')[-1]
            b = b.split('/')[-1].split('_')[0].split('T')[-1]
            if float(a) > float(b):
                csv_dir.append(temp_csv[0])
            else:
                csv_dir.append(temp_csv[1])
    return csv_dir


if __name__ == "__main__":
    is_parallel = True  # 是否并行
    csv_dir = "/data/data1/0923_STI_Dataset/0923_STI/Dev/0923_STI.csv"
    print('csvdir:', csv_dir)
    if not is_parallel:
        # 串行
        for i in range(len(dir_str)):
            # 仔细检查，千万不能调用自己！！！！！！
            command = "python 0929_addNoiseNSplitWav.py --csv_file " + csv_dir[i] + " --dir_str " + dir_str[
                i] + " --save_dir " + \
                      save_dir[i]
            os.system(command)
    else:
        # 并行
        commands = ["python 0929_addNoiseNSplitWav.py --csv_file " + csv_dir + " --dir_str " + dir_str[i] + " --save_dir " + save_dir[i] for i in range(len(dir_str))]
        print('commands:', commands)
        threads = []
        for cmd in commands:
            th = threading.Thread(target=execCmd, args=(cmd,))
            th.start()
            threads.append(th)
        # 等待线程运行完毕
        for th in threads:
            th.join()

