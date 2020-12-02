import sys
sys.path.append("..")
from general import *
from general.data import *
from general.all_imports import *
from general.pl_callbacks import EpochChange

from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

class AutoEncoderLM(LightningModule):
    def __init__(self,
                 model,
                 save_path,
                 loss_type,
                 optimizer_type,
                 default_learning_rate = 1e-3,
                 max_epochs = 300,
                 status_rate = 25,
                 lr_scheduler_kwargs = {
                     'factor': 0.75,
                     'patience': 5,
                     'threshold': 1e-4,
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
        
        self.epoch_train_loss = list()
        self.epoch_validation_loss = list()

        if "vae" in self.model_file.lower():
            self.step = self.vae_step
        self.addNoise = False
        if "nois" in self.model_file.lower():
            self.addNoise = True
        if "patch" in self.model_file.lower():
            self.step = self.patch_step
        if "stack" in self.model_file.lower():
            self.step = self.stack_step
        
    def epoch_reset(self,):
        train_loss = np.mean(self.epoch_train_loss)
        val_loss = np.mean(self.epoch_validation_loss)
        self.history["train_loss"].append(train_loss)
        self.history["validation_loss"].append(val_loss)
        self.epoch_train_loss = list()
        self.epoch_validation_loss = list()
        
    def epoch_status(self,):
        print("="*60)
        print("Epoch: [%03d/%03d] | Time: %0.2f (s) | Model: %s"%(self.EPOCH, self.MAX_EPOCHS, (self.EPOCH_END_TIME-self.EPOCH_START_TIME), self.model_file))
        print("-"*60)
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
        print(pd.DataFrame(d).T)
#         print("="*60)
        
    def save(self,):
        save_model(self.model, self.model_file)
    
    def save_final(self,):
        self.save()
        plot_stat(self.history, os.path.split(self.model_file)[-1], self.save_path)
        with open(os.path.join(self.save_path, "train_stats.pkl"), "wb") as f:
            pkl.dump(self.history, f)

    def get_inputs(self, images):
        if self.addNoise:
            return add_noise(images)
        return images

    def step(self, images):
        reconstructions, encodings = self.model(self.get_inputs(images))
        return self.loss_criterion(images, reconstructions)

    def patch_step(self, images):
        patch_images = get_patches(images)
        reconstructions, encodings = self.model(patch_images)
        return self.loss_criterion(patch_images, reconstructions)

    def stack_step(self, images):
        # images 32x1x16x128x128
        stacked_images = images.squeeze(dim = -4)
        reconstructions, encodings = self.model(stacked_images)
        return self.loss_criterion(stacked_images, reconstructions)

    def vae_step(self, images):
        reconstructions, latent_mu, latent_logvar = self.model(self.get_inputs(images))
        return self.vae_loss(images, reconstructions, latent_mu, latent_logvar)      
    
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