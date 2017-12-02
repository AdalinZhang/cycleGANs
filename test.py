import time
import os
from .options.test_options import TestOptions
from .data.data_loader import CreateDataLoader
from .models.models import create_model
from .utilSet.visualizer import Visualizer
from .utilSet import html

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

if __name__ == '__main__':
    # create website
    web_dir = os.path.join('./results/', 'experiment_name', 'test_latest')
    webpage = html.HTML(web_dir, 'Experiment = cycleGANs, Phase = test, Epoch = latest')
    # test
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()

        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)

    webpage.save()
