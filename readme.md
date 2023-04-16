## 运行说明

- 如果要加新的房间混响

  - 首先需要生成一批【不含噪，只卷积了混响】的wav文件

    1. 这个过程同T60数据生成的第一步
    2. 如果这一步中加了非TIMIT的人声，需要在`0929_addNoiseNSplitWav.py`做以下修改
       - 将你使用的干净人声wav放到同一个文件夹下，然后将第44行的default修改为此文件夹路径。
       - 将第105行，`people_choose`对应的 list 改为所有你用到的人声名称
         1. 如果觉得麻烦，直接把`is_file_people_choose()`函数修改成 return True也行

  - 对使用到的房间RIR进行一个开始时长统计，结果保存到`cutTimeDict.py`中

    1. 把用到的所有RIR wav文件存在同一文件夹下，在`calcutTime.py`中修改`original_file_head`为 这个文件夹的路径，然后运行`calcutTime.py`，会在代码目录下生成一个pkl文件，里面是一个dict
    2. 把这个dict复制到`cutTimeDict.py`，并覆盖掉之前的`rir_dict`
       - 也可以保存到其他的.py文件里，确保dict变量名为`rir_dict`，然后在`0929_addNoiseNSplitWav.py`中修改第28行，from 你的.py文件 import rir_dict

  - 在`thread_0929.py`修改以下两个变量

    1. `dir_str_head`：第一步生成的wav文件所在的目录

    2. `save_dir_head`：生成的STI pt文件的保存目录

       

  - 如果要加新噪声

    - 参考`/data/xbj/all_noise_all.txt`格式，将需要加的噪声的路径放在一个txt文件里

    - 在`0929_addNoiseNSplitWav.py`中修改以下parse_argument的default值：

      1. `noise_txt`：修改为你新加噪声的txt路径

         

  - 如果要改变信噪比

    - 在`0929_addNoiseNSplitWav.py`第332行，修改 list 中元素为新的信噪比

      

  - 在上述修改均完成后，运行`thread_0929.py`即可

  - 运行完之后需要手动将一些房间分出来作为validation

    - 具体做法就是`mv ./train/some_config/*val_room_name*.pt `` ``./val/some_config/`