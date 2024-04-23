import matplotlib.pyplot as plt
import os
import numpy as np

fi_1 = open('YOLOX_outputs/yolox_voc_s/train_log.txt', 'r', encoding='utf-8')  # 1、读取路径,改为自己的路径
iters_num = int('63300')  # 2、自己估计下坐标轴x,这里是总迭代次数

lines = fi_1.readlines()

total_loss = []
iou_loss = []
l1_loss = []
conf_loss = []
cls_loss = []
for line in lines:
    if 'total_loss' in line:
        print(line)
        line0 = line.split('total_loss: ')[-1].split(', iou_loss:')[0]
        line1 = line.split('iou_loss: ')[-1].split(', l1_loss:')[0]
        line2 = line.split('l1_loss: ')[-1].split(', conf_loss:')[0]
        line3 = line.split('conf_loss: ')[-1].split(', cls_loss:')[0]
        line4 = line.split('cls_loss: ')[-1].split(', lr:')[0]
        total_loss.append(float(line0))
        iou_loss.append(float(line1))
        l1_loss.append(float(line2))
        conf_loss.append(float(line3))
        cls_loss.append(float(line4))
        print('-----------', line0, line1, line2, line3, line4)
        # break
print(len(total_loss))
# plt.style.use('ggplot')
plt.rc('font', family='Times New Roman', size=13)  # 全局中英文为字体“罗681马字体”
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.figure(0, figsize=(10, 8))
x = np.arange(0, iters_num, 10)  ################################ 自己估计下坐标轴x,这里10是源代码默认iter=10输出一次loss
plt.subplot(2, 3, 1)
plt.plot(x, total_loss, color='blue', label="Total Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.grid(True)
plt.legend(loc="upper right", fontsize='xx-small')
plt.subplot(2, 3, 2)
plt.plot(x, iou_loss, color='blue', label="eiou Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.grid(True)
plt.legend(loc="upper right", fontsize='xx-small')
plt.subplot(2, 3, 3)
plt.plot(x, l1_loss, color='green', label="l1 Loss")
plt.xlabel("Steps")
plt.ylabel("")
plt.grid(True)
plt.legend(loc="upper right", fontsize='xx-small')
plt.subplot(2, 3, 4)
plt.plot(x, conf_loss, color='darkorange', label="conf Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.grid(True)
plt.legend(loc="upper right", fontsize='xx-small')
plt.subplot(2, 3, 5)
plt.plot(x, cls_loss, color='indigo', label="cls Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.grid(True)
plt.legend(loc="upper right", fontsize='xx-small')
plt.annotate("Loss", (-2, 10), xycoords='data', xytext=(-2, 10), fontsize=15)
plt.savefig('losses.png', dpi=600, bbox_inches='tight')
plt.show()