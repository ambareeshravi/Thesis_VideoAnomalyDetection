from pytorch_lightning.callbacks import Callback
from time import time
from .utils import eta

class EpochChange(Callback):
    def __init__(self):
        pass
    
    def on_epoch_start(self, trainer, pl_module):
        pl_module.EPOCH_START_TIME = time()
              
    def on_epoch_end(self, trainer, pl_module):
        pl_module.EPOCH_END_TIME = time()
        pl_module.epoch_reset()
        
        if pl_module.EPOCH == 0:
            pl_model.cl.print(eta(pl_module.EPOCH, pl_module.MAX_EPOCHS, (pl_module.EPOCH_END_TIME-pl_module.EPOCH_START_TIME)))
            
        if pl_module.EPOCH % pl_module.status_rate == 0:
            pl_module.epoch_status()
            pl_module.save()
        pl_module.EPOCH += 1

    def on_train_end(self, trainer, pl_module):
        pl_module.save_final()