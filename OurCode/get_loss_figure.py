import matplotlib.pyplot as plt
import re
from mpl_toolkits.axes_grid1 import host_subplot
import numpy as np


file = '../checkpoints/experiment_name/loss_log.txt'

'''
获取loss数据存储到集合中
输入：文件夹路径
输出：数据列表，（嵌套列表）
'''
def get_loss_data(dir):
    data = []
    data_list = []
    fp = open(dir, 'r')
    for ln in fp:
        if 'epoch: ' in ln:
            #eopch
            arr = re.findall(r'epoch: \b\d+\b', ln)
            epoch_data = int(arr[0].strip(' ')[7:])
            data.append(epoch_data)
            #iters
            arr1 = re.findall(r'iters: \b\d+\b', ln)
            iters_data = int(arr1[0].strip(' ')[7:])
            data.append(iters_data)
            #D_A
            arr2 = re.findall(r'D_A: \b\S+\b', ln)
            D_A_data = float(arr2[0].strip(' ')[5:])
            data.append(D_A_data)
            #G_A
            arr3 = re.findall(r'G_A: \b\S+\b', ln)
            D_A_data = float(arr3[0].strip(' ')[5:])
            data.append(D_A_data)
            #Cyc_A
            arr4 = re.findall(r'c_A: \b\S+\b', ln)
            D_A_data = float(arr4[0].strip(' ')[5:])
            data.append(D_A_data)
            #D_B
            arr5 = re.findall(r'D_B: \b\S+\b', ln)
            D_A_data = float(arr5[0].strip(' ')[5:])
            data.append(D_A_data)
            #G_B
            arr6 = re.findall(r'G_B: \b\S+\b', ln)
            D_A_data = float(arr6[0].strip(' ')[5:])
            data.append(D_A_data)
            #Cyc_B
            arr7 = re.findall(r'c_B: \b\S+\b', ln)
            D_A_data = float(arr7[0].strip(' ')[5:])
            data.append(D_A_data)
            #idt_A
            arr8 = re.findall(r't_A: \b\S+\b', ln)
            D_A_data = float(arr8[0].strip(' ')[5:])
            data.append(D_A_data)
            #idt_B
            arr9 = re.findall(r't_B: \b\S+\b', ln)
            D_A_data = float(arr9[0].strip(' ')[5:])
            data.append(D_A_data)

            data_list.append(data)
        data = []
    fp.close()
    return data_list

'''
获取每轮loss平均值
输入:数据列表，loss选择，训练的数据量
输出:loss的avr列表
'''
def get_all_avr_loss(data, which, iters):
    sum = 0
    avr_list = []
    count = 0
    #D_A, G_A, Cyc_A, D_B, G_B, Cyc_B, idt_A, idt_B
    if which == 'D_A':
        for var in data:
            sum += data[count][2]   # D_A在列表中的第3位
            count += 1
            if count % iters == 0:
                avr_list.append(sum / iters)
                sum = 0
    elif which == 'G_A':
        for var in data:
            sum += data[count][3]   # G_A在列表中的第4位
            count += 1
            if count % iters == 0:
                avr_list.append(sum / iters)
                sum = 0
    elif which == 'Cyc_A':
        for var in data:
            sum += data[count][4]  # Cyc_A在列表中的第5位
            count += 1
            if count % iters == 0:
                avr_list.append(sum / iters)
                sum = 0
    elif which == 'D_B':
        for var in data:
            sum += data[count][5]  # D_B在列表中的第6位
            count += 1
            if count % iters == 0:
                avr_list.append(sum / iters)
                sum = 0
    elif which == 'G_B':
        for var in data:
            sum += data[count][6]  # G_B在列表中的第7位
            count += 1
            if count % iters == 0:
                avr_list.append(sum / iters)
                sum = 0
    elif which == 'Cyc_B':
        for var in data:
            sum += data[count][7]  # Cyc_B在列表中的第8位
            count += 1
            if count % iters == 0:
                avr_list.append(sum / iters)
                sum = 0
    elif which == 'idt_A':
        for var in data:
            sum += data[count][8]  # idt_A在列表中的第9位
            count += 1
            if count % iters == 0:
                avr_list.append(sum / iters)
                sum = 0
    elif which == 'idt_B':
        for var in data:
            sum += data[count][9]  # idt_B在列表中的第10位
            count += 1
            if count % iters == 0:
                avr_list.append(sum / iters)
                sum = 0
    # print(avr_list)

    return avr_list
'''
画图函数
输入：选择要画的是哪个数据、保存的name(str)、输入图片数量
输出：保存为id.png
'''
def draw_loss_figure(which, iters):
    data = get_loss_data(file)#获得原始数据列表
    avr = get_all_avr_loss(data, which, iters)#获得计算后的平均值
    # print(avr)

    host = host_subplot(1, 1, 1)  # 分几个图，选择第几个图
    plt.subplots_adjust(right=0.8)   # 限定右边界

    host.set_xlabel("epoch")
    host.set_ylabel("%s loss" % which)

    p1, = host.plot(avr, label="train cycleGANs %s loss" % which)

    host.legend(loc=1)  # 标签距离边界位置

    host.axis["left"].label.set_color(p1.get_color())

    host.set_xlim([-1, len(avr)])    # x轴范围
    host.set_ylim([0., np.max(avr)])  # y轴范围

    plt.draw()
    plt.savefig('./getLossFigure/loss_Img/%s.png' % which)
    # plt.show()


if __name__ == '__main__':
    # [epoch, iters, D_A, G_A, Cyc_A, D_B, G_B, Cyc_B, idt_A, idt_B]
    draw_loss_figure('D_B', 500)  # 要生成图片的参数名，训练的图片数量
