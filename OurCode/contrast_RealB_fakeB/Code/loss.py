import ntpath
import os

from .save_loss import SaveLoss

if __name__ == '__main__':
    loss = SaveLoss()

    # 图片路径
    imgpath = "checkpoints/experiment_name/web/images"
    # 获取文件目录下包含指定字符串的图片名路径
    fake_B = loss.get_path_list(imgpath, "fake_B")
    real_B = loss.get_path_list(imgpath, "real_B")

    for i in range(len(fake_B)):
        fake_path = ''.join(fake_B[i])   # ''.join()获取list的字符串值
        real_path = ''.join(real_B[i])
        img = loss.get_loss_img(fake_path, real_path)

        # get image name
        short_path = ntpath.basename(fake_path)
        name = os.path.splitext(short_path)[0]
        # img.show()
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save("../Img/%s.jpg" % name)

        # write image loss  (img_loss.txt)
        loss.print_img_loss(name)
