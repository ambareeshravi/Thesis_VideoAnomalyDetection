import sys
sys.path.append("..")
from general import *
from general.data import *
from general.all_imports import *
from general.pl_callbacks import EpochChange
from autoencoder_steps import AutoEncoderHelper

from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, GPUStatsMonitor

class AutoEncoderLM(LightningModule, AutoEncoderHelper):
    def __init__(self,
                 model,
                 save_path,
                 loss_type,
                 optimizer_type,
                 noise_var = 0.1,
                 default_learning_rate = 1e-3,
                 max_epochs = 300,
                 status_rate = 25,
                 lr_scheduler_kwargs = {
                     'factor': 0.75,
                     'patience': 4,
                     'threshold': 5e-5,
                     'verbose': True
                 }
                ):
        super().__init__()
        # General params
        self.learning_rate = default_learning_rate
        self.model = model
        self.loss_criterion = select_loss[loss_type]
        self.optimizer_type = optimizer_type
        self.status_rate = status_rate
        self.EPOCH = 0
        self.MAX_EPOCHS = max_epochs
        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        
        
        # Path
        self.save_path = save_path
        create_directory(self.save_path)
        self.model_file = getModelFileName(self.save_path)
        
        # Model Params
        self.history = {
            "train_loss": list(),
            "validation_loss": list()
        }
        
        self.cl = CustomLogger(join_paths([self.save_path, "train_logs"]))
        
        self.epoch_train_loss = list()
        self.epoch_validation_loss = list()
        AutoEncoderHelper.__init__(self, model_file = self.model_file, noise_var = noise_var)
               
    def epoch_status(self,):
        if self.EPOCH == 1: print(execute_bash("nvidia-smi"))
        cl.print("="*60)
        cl.print("Epoch: [%03d/%03d] | Time: %0.2f (s) | Model: %s"%(self.EPOCH, self.MAX_EPOCHS, (self.EPOCH_END_TIME-self.EPOCH_START_TIME), self.model_file))
        cl.print("-"*60)
        try:
            d = {
            "Training" : {"Loss -2": self.history["train_loss"][-3],
                        "Loss -1": self.history["train_loss"][-2],
                        "Loss *" : self.history["train_loss"][-1],
                         },
            "Validation" : {"Loss -2": self.history["validation_loss"][-3],
                            "Loss -1": self.history["validation_loss"][-2],
                            "Loss *" : self.history["validation_loss"][-1],
                         },
        }
        except:
            d = {
            "Training" : {"Loss" : self.history["train_loss"][-1],},
            "Validation" : {"Loss" : self.history["validation_loss"][-1],},
            }
        cl.print(pd.DataFrame(d).T)
#         cl.print("="*60)
          
    
    # LM functions
    def configure_optimizers(self):
        optimizer = select_optimizer[self.optimizer_type](self.model, lr = self.learning_rate)
        return {
           'optimizer': optimizer,
           'lr_scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
               optimizer,
               **self.lr_scheduler_kwargs
           ),
           'monitor': 'validation_loss'
       }

    def training_step(self, batch, batch_idx):
        images, labels = batch
        train_loss = self.step(images)
        self.epoch_train_loss.append(train_loss.item())
        self.log('training_loss', train_loss, on_epoch=True, logger=True, sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        val_loss = self.step(images)
        self.epoch_validation_loss.append(val_loss.item())
        self.log('validation_loss', val_loss, on_epoch=True, logger=True, sync_dist=True)
        return val_loss