import sys
sys.path.append("..")
from general.all_imports import *
from general.model_utils import *
from general.data import *
from general.utils import *

from C2D_Models import *
from C3D_Models import *
from ConvLSTM_AE.ConvLSTM_AE import *
from PatchWise.models_PatchWise import PatchWise_C2D

from universal_tester import *

from SVM.deep_SVDD import DeepSVDD
        
class SVDD_FineTuner:
    def __init__(
        self,
        model,
        model_path: str,
        dataset_type: str,
        lr = 5e-7,
        batch_size = 256,
        epochs = [2,5,10,20],
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
        
        self.test_data, channels = select_dataset(dataset_type, isTrain = False, sample_stride = 1, asImages = True, **dataset_kwargs)
        self.tester = AutoEncoder_Tester(
                        model = self.model,
                        dataset = self.test_data,
                        model_file = self.model_path,
                        save_vis = False,
                    )

        
        if testOriginal: self.tester.test()

        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        if isinstance(self.epochs, int): self.epochs = [self.epochs]
        
        self.device = torch.device("cpu")
        if useGPU and torch.cuda.is_available():
            self.device = torch.device("cuda")
            INFO("Device set to GPU")
        self.model.to(self.device)
        
        self.train_data, channels = select_dataset(dataset_type, **dataset_kwargs)
        INFO("Created Dataset")
        
        self.initial_train_encodings = self.get_train_encodings()
        INFO("Created Encodings")
        
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
        INFO("SVDD Ready")
        
        self.train_loader, self.val_loader = get_data_loaders(self.train_data, batch_size=128)
        INFO("Data Loaders Ready. Starting the training")
        
        for idx, e in enumerate(self.epochs):
            print("="*40)
            print("Testing: For epochs =", e, "and model:", self.model_path)
            if idx ==0: epochs = e
            else: epochs = abs(self.epochs[idx] - epochs) # change
            self.fit(epochs, original_epochs = e)
            INFO("Training Completed. Testing the model now...")
            
            self.tester.model = self.model
            self.tester.model_file = ("_SVDD%d.pth.tar"%(e)).join(self.model_path.split(".pth.tar"))
            self.tester.model.to(self.tester.device)
            self.tester.model.eval()
            self.tester.test()
            print("="*40)
        
    def get_train_encodings(self):
        encodings = list()
        self.model.eval()
        for image in tqdm(self.train_data.data):
            with torch.no_grad():
                e = self.model.encoder(image.unsqueeze(dim = 0).to(self.device))
                encodings.append(e.detach().flatten(start_dim = 1, end_dim = -1).cpu())
        encodings = torch.cat(encodings)
        return encodings
    
    def fit(self, epochs, original_epochs):
        train_loss_list, val_loss_list = list(), list()

        for epoch in range(epochs):
            epoch_st = time()
            epoch_train_loss, epoch_val_loss = list(), list()
            self.model.train()
            for batch_train_idx, batch_train_data in enumerate(self.train_loader):
                train_images, train_labels = batch_train_data
                train_encodings = self.model.encoder(train_images.to(self.device))
                train_loss = self.svdd.train_step(train_encodings)
                epoch_train_loss.append(train_loss.item())

            self.model.eval()
            for batch_val_idx, batch_val_data in enumerate(self.val_loader):
                val_images, val_labels = batch_val_data
                with torch.no_grad():
                    val_encodings = self.model.encoder(val_images.to(self.device))
                    val_loss = self.svdd.val_step(val_encodings)
                    epoch_val_loss.append(val_loss.item())
            epoch_train_loss = np.mean(epoch_train_loss)
            epoch_val_loss = np.mean(epoch_val_loss)

            train_loss_list.append(epoch_train_loss)
            val_loss_list.append(epoch_val_loss)
            print("Epoch [%d/%d] | Train Loss: %0.6f | Val Loss: %0.6f | Time: (%d) s"%(epoch, epochs, epoch_train_loss, epoch_val_loss, (time()-epoch_st)))
        
        save_model(self.model, ("_SVDD%d.pth.tar"%(original_epochs)).join(self.model_path.split(".pth.tar")))