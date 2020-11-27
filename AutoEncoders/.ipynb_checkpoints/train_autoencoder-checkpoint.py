import sys
sys.path.append("..")
from general import *
from general.model_utils import HalfPrecision, DataParallel

class AutoEncoderModel:
    def __init__(self,
                 model,
                 save_path,
                 loss_criterion,
                 optimizer,
                 device = torch.device("cuda"),
                 lr_scheduler_params = {"factor": 0.75, "patience": 5, "threshold": 1e-4},
                 useHalfPrecision = False,
                 debug = True
                ):
        self.isHalfPrecision = useHalfPrecision
        self.model = model
        if self.isHalfPrecision:
            HalfPrecision(self.model)
            
        self.model = DataParallel(self.model)
            
        self.device = device
        self.lr_scheduler_params = lr_scheduler_params
        self.debug = debug
        
        # Path
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.model_file = join_paths([self.save_path, os.path.split(self.save_path)[-1] + ".pth.tar"])
        if "vae" in self.model_file.lower():
            self.step = self.vae_step
        self.addNoise = False
        if "nois" in self.model_file.lower():
            self.addNoise = True
        if "patch" in self.model_file.lower():
            self.step = self.patch_step
        if "stack" in self.model_file.lower():
            self.step = self.stack_step
        
        # Model Params
        self.stopTraining = False
        self.history = {
            "train_loss": list(),
            "validation_loss": list()
        }
        
        self.epoch_train_loss = list()
        self.epoch_validation_loss = list()
        
        self.model.to(self.device)
        
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = self.lr_scheduler_params["factor"], patience = self.lr_scheduler_params["patience"], threshold = self.lr_scheduler_params["threshold"])
        with open(os.path.join(self.save_path, "model.txt"), "w") as f:
            f.write(str(self.model))
        
    def epoch_reset(self,):
        train_loss = np.mean(self.epoch_train_loss)
        val_loss = np.mean(self.epoch_validation_loss)
        self.history["train_loss"].append(train_loss)
        self.history["validation_loss"].append(val_loss)
        self.lr_scheduler.step(val_loss)
        self.epoch_train_loss = list()
        self.epoch_validation_loss = list()
        
    def epoch_status(self,):
        if not self.stopTraining:            
            print("Model:", self.model_file)
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
            print("-"*40)
        
    def save(self,):
        save_model(self.model, self.model_file)
    
    def save_final(self,):
        self.save()
        plot_stat(self.history, os.path.split(self.model_file)[-1], self.save_path)
        with open(os.path.join(self.save_path, "train_stats.pkl"), "wb") as f:
            pkl.dump(self.history, f)             
    
    def get_inputs(self, images):
        if self.addNoise:
            return add_noise(images, device = self.device)
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
        return self.model.vae_loss(images, reconstructions, latent_mu, latent_logvar)
        
    def train_step(self, images):
        self.model.to(self.device)
        self.model.train()
        self.model.zero_grad()
        if self.isHalfPrecision: images = images.half()
        loss = self.step(images)
        loss.backward()
        self.optimizer.step()
        self.model.to('cpu')
        self.epoch_train_loss.append(loss.item())
        
    def val_step(self, images):
        self.model.to(self.device)
        self.model.eval()
        if self.isHalfPrecision: images = images.half()
        with torch.no_grad():
            loss = self.step(images)
        self.model.to('cpu')
        self.epoch_validation_loss.append(loss.item())
        
    def self_destruct(self):
        try: del self.model, self.optimizer, self.loss_criterion, self.history
        except: pass

class AutoEncoder_Trainer:
    def __init__(self,
                 models_list,
                 model_paths,
                 train_loader,
                 val_loader,
                 optimizer,
                 loss_criterion,
                 epochs = 100,
                 status_rate = 1,
                 lr_scheduler_params = {"factor": 0.9, "patience": 5, "threshold": 5e-4},
                 useHalfPrecision = False,
                 destructAll = True,
                 useGPU = True,
                 debug = True):
        
        self.epochs = epochs
        self.status_rate = status_rate
        self.device = torch.device("cpu")
        if useGPU and torch.cuda.is_available(): self.device = torch.device("cuda")
        self.debug = debug
        self.destructAll = destructAll
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        if not isinstance(models_list, list): models_list = [models_list]
        if not isinstance(model_paths, list): model_paths = [model_paths]*len(models_list)
        if not isinstance(loss_criterion, list): loss_criterion = [loss_criterion]*len(models_list)
        if not isinstance(optimizer, list): optimizer = [optimizer]*len(models_list)
                
        self.autoencoder_models = list()
        
        for idx in range(len(models_list)):
            model = AutoEncoderModel(
                     models_list[idx],
                     model_paths[idx],
                     loss_criterion[idx],
                     optimizer[idx],
                     device = self.device,
                     lr_scheduler_params = lr_scheduler_params,
                     useHalfPrecision = useHalfPrecision,
                     debug = debug
                    )                            
            self.autoencoder_models.append(model)
            
    def train(self):
        for epoch in range(1, self.epochs + 1):
            epoch_st = time()
            
            # Train
            for train_batch_idx, train_batch in enumerate(self.train_loader):
                train_images, train_labels = train_batch
                train_images = train_images.to(self.device)
                for model in self.autoencoder_models:
                    if model.stopTraining: continue
                    model.train_step(train_images)
                
            # Validation
            if self.val_loader != None:
                for val_batch_idx, val_batch in enumerate(self.val_loader):
                    val_images, val_labels = val_batch
                    val_images = val_images.to(self.device)
                    for model in self.autoencoder_models:
                        if model.stopTraining: continue
                        model.val_step(val_images)
            
            # Print Epoch stats
            print_status = epoch % self.status_rate == 0 or epoch == 1
            if print_status:
                if epoch == 1: eta(epoch, self.epochs, (time() - epoch_st))
                print("-"*60)
                print("Epoch: [%03d/%03d] | time/epoch: %0.2f seconds"%(epoch, self.epochs, (time() - epoch_st)))
                print("-"*60)
                
            for model in self.autoencoder_models:
                if model.stopTraining: continue
                model.epoch_reset()
                if print_status: model.epoch_status()
        
        # Finally save models
        model_paths = list()
        for model in self.autoencoder_models:
            model.save_final()
            model_paths.append(model.save_path)
        
        # Clear memory
        try:
            if self.destructAll:
                for model in self.autoencoder_models:
                    model.self_destruct()
                del self.autoencoder_models
            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            print("Could not clear the memory. Kill the process manually.")
            print(e)
            
        return model_paths