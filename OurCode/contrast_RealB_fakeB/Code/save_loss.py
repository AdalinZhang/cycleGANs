import os
import time
import numpy as np
from PIL import Image


class SaveLoss():
    def __init__(self):
        self.size = 256
        self.log_name = os.path.join('../txt', 'img_loss.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Image Loss (%s) ================\n' % now)

    def imgtotensor(self, filename):
        # read image
        img = Image.open(filename).convert("L")
        data = np.matrix(img.getdata(), dtype='float') / 255.0
        new_data = np.reshape(data, img.size)
        # print(new_data.shape)
        return new_data

    def get_avg_loss(self):
        loss = None
        loss = np.sum(np.abs(self.real - self.fake)) / (self.size * self.size) * 100
        return loss

    def get_loss_img(self, fakeimg, realimg):
        self.real = self.imgtotensor(realimg)
        self.fake = self.imgtotensor(fakeimg)
        loss = np.abs(self.real - self.fake)
        # matrix to image
        data = np.matrix(loss, dtype='float') * 255.0
        self.img = Image.fromarray(data)
        return self.img

    # print loss value
    def print_img_loss(self, img):
        message = '(image: %s,  loss_percentage : %.2f%% ) ' % (img, self.get_avg_loss())
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def get_path_list(self, file_dir, name):
        L = []
        for root, dirs, files in os.walk(file_dir):
            for file in sorted(files):    # 遍历文件目录下每一个文件
                if name in file:  # 判断是否包含指定字符串
                    L.append(os.path.join(root, file))
        return L