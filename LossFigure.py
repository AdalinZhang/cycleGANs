#!/usr/bin/env python
import matplotlib.pyplot as plt
import re
from mpl_toolkits.axes_grid1 import host_subplot

# read the log file

file = './checkpoints/experiment_name/loss_log.txt'

'''
读取数据
'''
def get_loss(dir):
    train_loss = []
    fp = open(dir, 'r')
    for ln in fp:
        if ') D_A: ' in ln:
            arr = re.findall(r'D_A: \b\S+\b', ln)
            # print(arr)
            train_loss.append(arr[0].strip(' ')[6:])
            # print(train_loss)

    data_numpy = len(train_loss)

    fp.close()

    return train_loss,data_numpy

'''
画图
'''
def draw_loss_figure(save_id):
    host = host_subplot(1,1,1)#分几个图，选择第几个图
    plt.subplots_adjust(right=0.8)# 限定右边界

    host.set_xlabel("epoch")
    host.set_ylabel("D_A loss")

    train_loss,data_numpy = get_loss(file)

    p1, = host.plot(train_loss, label="train cycleGANs D_A loss")

    host.legend(loc=1)#标签距离边界位置

    host.axis["left"].label.set_color(p1.get_color())

    host.set_xlim([-1, data_numpy])#x轴范围
    host.set_ylim([0.,1])#y轴范围

    plt.draw()
    plt.savefig('./lossfigure/%d.png'%save_id)
    #plt.show()

if __name__ == '__main__':
    draw_loss_figure(1)