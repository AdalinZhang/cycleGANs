
#模型的外部接口，简化成只使用cycleGANModel()
def create_model(opt):
    #print(opt.model)

    from .cycle_gan_model import CycleGANModel
    model = CycleGANModel()
    model.initialize(opt)
    #print("model [%s] was created" % (model.name()))
    return model
