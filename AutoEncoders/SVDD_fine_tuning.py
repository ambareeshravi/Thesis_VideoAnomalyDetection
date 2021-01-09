import torch
from torch import nn
import sys
sys.path.append("..")
from general.model_utils import *
from general.data import *
from general.utils import *

from C2D_Models import *
from C3D_Models import *
from ConvLSTM_AE.ConvLSTM_AE import *
from PatchWise.models_PatchWise import PatchWise_C2D

from universal_tester import test_model

from SVM.deep_SVDD import DeepSVDD
        
class SVDD_FineTuner:
    def __init__(
        self,
        model,
        model_path: str,
        dataset_type: str,
        lr = 5e-7,
        batch_size = 256,
        epochs = 10,
        dataset_kwargs = {
            "image_size": 128,
            "image_type": "normal",
        },
        tester_kwargs = {
                "stackFrames": 1,
                "save_vis": False,
                "n_seed": 8,
                "useGPU": True
            },
        testOriginal = False,
        useGPU = True
    ):
        self.model = model
        self.model_path = model_path
        self.dataset_type = dataset_type
        
        if testOriginal:
            test_model(
                dataset_type = self.dataset_type,
                model = self.model,
                model_path = self.model_path,
                tester_kwargs = **tester_kwargs,
        )

        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.device = torch.device("cpu")
        if useGPU and torch.cuda.is_available():
            self.device = torch.device("cuda")
        self.model.to(self.device)
        
        self.train_data, channels = select_dataset(dataset_type, **dataset_kwargs)
        self.initial_train_encodings = self.get_train_encodings()
        self.svdd = DeepSVDD()
        self.svdd.set_trainer(
            self.model,
            self.initial_train_encodings,
            lr = lr,
            weight_decay = 1e-7,
            lr_scheduler_kwargs = {
                "factor": 0.5,
                "patience": 4,
                "threshold": 1e-4,
                "min_lr": 0,
                "eps": 1e-08,
                "verbose": True,
            }
        )
        self.svdd_train_loader, self.svdd_val_loader = get_data_loaders(self.train_data, batch_size=128)
        self.fit()
        test_model(
            dataset_type = self.dataset_type,
            model = self.model,
            model_path = ("_SVDD%d.pth.tar"%(self.epochs)).join(self.model_path.split(".pth.tar")),
            tester_kwargs = **tester_kwargs,
        )
        
    def get_train_encodings(self):
        encodings = list()
        self.model.eval()
        for image in tqdm(self.train_data.data):
            with torch.no_grad():
                e = self.model.encoder(image.unsqueeze(dim = 0))
                encodings.append(e.detach().flatten(start_dim = 1, end_dim = -1))
        encodings = torch.cat(encodings)
        return encodings
    
    def fit(self):
        train_loss_list, val_loss_list = list(), list()

        for epoch in range(self.epochs):
            epoch_st = time()
            epoch_train_loss, epoch_val_loss = list(), list()
            self.model.train()
            for batch_train_idx, batch_train_data in enumerate(self.train_loader):
                train_images, train_labels = batch_train_data
                train_encodings = self.model.encoder(train_images.to(device))
                train_loss = self.svdd.train_step(train_encodings)
                epoch_train_loss.append(train_loss.item())

            self.model.eval()
            for batch_val_idx, batch_val_data in enumerate(self.val_loader):
                val_images, val_labels = batch_val_data
                with torch.no_grad():
                    val_encodings = self.model.encoder(val_images.to(device))
                    val_loss = self.svdd.val_step(val_encodings)
                    epoch_val_loss.append(val_loss.item())
            epoch_train_loss = np.mean(epoch_train_loss)
            epoch_val_loss = np.mean(epoch_val_loss)

            train_loss_list.append(epoch_train_loss)
            val_loss_list.append(epoch_val_loss)
            print("Epoch [%d/%d] | Train Loss: %0.6f | Val Loss: %0.6f | Time: (%d) s"%(epoch, EPOCHS, epoch_train_loss, epoch_val_loss, (time()-epoch_st)))
        
        save_model(self.model, ("_SVDD%d.pth.tar"%(self.epochs)).join(self.model_path.split(".pth.tar")))