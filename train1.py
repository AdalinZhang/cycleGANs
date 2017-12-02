import time
from .options.train_options import TrainOptions
from .data.data_loader import CreateDataLoader
from .models.models import create_model
from .utilSet.visualizer import Visualizer

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

model = create_model(opt)
visualizer = Visualizer(opt)

#减少数据量，以及训练次数
#图片数量为1，训练轮数为1时可运行，可用于检查代码是否有误

if __name__ == '__main__':
    total_steps = 0
    epoch = 1

    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += 1
        epoch_iter += 1
        model.set_input(data)
        model.optimize_parameters()

        save_result = True
        visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        errors = model.get_current_errors()
        t = (time.time() - iter_start_time) / opt.batchSize
        visualizer.print_current_errors(epoch, epoch_iter, errors, t)
        # epoch 训练轮数  iters 训练图片数量，

        print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
        model.save('latest')


    print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
    print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, 200, time.time() - epoch_start_time))
    model.save('latest')
    model.save(epoch)

    model.update_learning_rate()#更新学习率
