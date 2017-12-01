import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from utilSet.visualizer import Visualizer
from LossFigure import draw_loss_figure

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)

model = create_model(opt)
visualizer = Visualizer(opt)

if __name__ == '__main__':
    total_steps = 0
    for epoch in range(1, 201):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += 1
            epoch_iter += 1
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % 100 == 0:
                save_result = total_steps % 1000 == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            #每次运行都计算loss值
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            # epoch 训练轮数  iters 训练图片数量，

            if total_steps % dataset_size == 0:
                draw_loss_figure(total_steps/dataset_size)##以训练轮数命名

            if total_steps % 5000 == 0:
                #每5000张图片保存一次，此处共2200张图片，故每两轮多600张时会保存一次
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save('latest')

        if epoch % 5 == 0:
            #所有图片训练每五轮输出一次，这里一共训练200轮
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, 200, time.time() - epoch_start_time))
            model.save('latest')
            model.save(epoch)

        model.update_learning_rate()#更新学习率
