from pytorch_lightning.callbacks import Callback
from time import time

class EpochChange(Callback):
    def __init__(self):
        pass
    
    def on_epoch_start(self, trainer, pl_module):
        pl_module.EPOCH_START_TIME = time()
              
    def on_epoch_end(self, trainer, pl_module):
        pl_module.EPOCH_END_TIME = time()
        pl_module.epoch_reset()
        
        if not pl_module.EPOCH:
            eta(pl_module.EPOCH, pl_module.MAX_EPOCHS, (pl_module.EPOCH_END_TIME-self.EPOCH_START_TIME))
            print("-"*60)
            
        if not (pl_module.EPOCH % pl_module.status_rate):
            pl_module.epoch_status()
            pl_module.save()
        pl_module.EPOCH += 1

    def on_train_end(self, trainer, pl_module):
        pl_module.save_final()