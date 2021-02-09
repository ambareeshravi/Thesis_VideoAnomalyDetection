import sys
sys.path.append("..")
from general.all_imports import *
from general import *
from general.model_utils import HalfPrecision, DataParallel
from autoencoder_steps import AutoEncoderHelper

class AutoEncoderModel(AutoEncoderHelper):
    def __init__(self,
                 model,
                 save_path,
                 loss_criterion,
                 optimizer,
                 noise_var = 0.1,
                 device = torch.device("cuda"),
                 lr_scheduler_params = {
                     "factor": 0.75,
                     "patience": 4,
                     "threshold": 5e-5,
                     'verbose': True
                 },
                 early_stopping_params = {
                     "threshold": 1e-6,
                     "patience": 8
                 },
                 useHalfPrecision = False,
                 debug = True
                ):
        
        self.isHalfPrecision = useHalfPrecision
        self.model = model
        if self.isHalfPrecision: HalfPrecision(self.model)
            
#         self.model = DataParallel(self.model)
            
        self.device = device
        self.lr_scheduler_params = lr_scheduler_params
        self.early_stopping_params = early_stopping_params
        self.stop_count = 0 
        self.debug = debug
        
        # Model Params
        self.stopTraining = False
        self.history = {
            "train_loss": list(),
            "validation_loss": list()
        }
        
        self.epoch_train_loss = list()
        self.epoch_validation_loss = list()
        
#         self.model.to(self.device)
        
        self.loss_criterion = loss_criterion
        self.optimizer = optimizer
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **self.lr_scheduler_params)
        
        # Path
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.model_file = join_paths([self.save_path, os.path.split(self.save_path)[-1] + ".pth.tar"])
        
        AutoEncoderHelper.__init__(self, model_file = self.model_file, noise_var = noise_var)
        
        with open(os.path.join(self.save_path, "model.txt"), "w") as f:
            f.write(str(self.model))
        
        self.cl = CustomLogger(join_paths([self.save_path, "train_logs"]))

    def epoch_status(self, epoch, epochs, epoch_st):
        if epoch == 1: print("Model:", self.model_file)
        if not self.stopTraining:
            if epoch == 1:
                self.cl.print(eta(epoch, epochs, (time() - epoch_st)))
                print(execute_bash("nvidia-smi"))
            self.cl.print("-"*60)
            self.cl.print("Epoch: [%03d/%03d] | time/epoch: %0.2f seconds"%(epoch, epochs, (time() - epoch_st)))
            self.cl.print("-"*60)
            
            self.cl.print("Model:", self.model_file)
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
            self.cl.print(pd.DataFrame(d).T)
            self.cl.print("-"*40)           
     
    def train_step(self, images):
        self.model.to(self.device)
        self.model.train()
        self.model.zero_grad()
        if self.isHalfPrecision: images = images.half()
        loss = self.step(images)
        loss.backward()
        self.optimizer.step()
        
        if self.isSVDD_enabled:
            encodings = self.model.encoder(self.get_inputs(images))
            if not self.svdd_init:
                self.svdd.set_trainer(self.model.encoder, encodings)
                self.svdd_init = True
            svdd_train_loss = self.svdd.train_step(encodings, self.svdd_init and (self.svdd_warmup_count > self.svdd.boundary_warm_up))
            self.svdd.history["epoch_train_loss"].append(svdd_train_loss.item())
            self.svdd_warmup_count += 1
            
        self.model.to('cpu')
        self.epoch_train_loss.append(loss.item())
        
    def val_step(self, images):
        self.model.to(self.device)
        self.model.eval()
        if self.isHalfPrecision: images = images.half()
        with torch.no_grad():
            loss = self.step(images)
            
            if self.isSVDD_enabled:
                encodings = self.model.encoder(images)
                svdd_val_loss = self.svdd.val_step(encodings)
                self.svdd.history["epoch_val_loss"].append(svdd_val_loss.item())
                if self.svdd_warmup_count % 50 == 0:
                    self.cl.print("*** SVDD Val Loss: %0.6f ***"%(svdd_val_loss.item()))
                    
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
                 epochs = 120,
                 status_rate = 20,
                 lr_scheduler_params = {
                     "factor": 0.75,
                     "patience": 4,
                     "threshold": 5e-5,
                     'verbose': True
                 },
                 early_stopping_params = {
                     "threshold": 1e-6,
                     "patience": 8
                 },
                 noise_var = 0.1,
                 useHalfPrecision = False,
                 run_status_file = "run_status.txt",
                 destructAll = True,
                 useGPU = True,
                 debug = True):
        
        self.epochs = epochs
        self.status_rate = status_rate
        self.device = torch.device("cpu")
        if useGPU and torch.cuda.is_available(): self.device = torch.device("cuda")
        self.debug = debug
        self.run_status_file = run_status_file
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
                     noise_var = noise_var,
                     lr_scheduler_params = lr_scheduler_params,
                     early_stopping_params = early_stopping_params,
                     useHalfPrecision = useHalfPrecision,
                     debug = debug
                    )                            
            self.autoencoder_models.append(model)
    
    def clear_memory(self):
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
            
    def train(self):
        stopTraining = False
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
                
            for model in self.autoencoder_models:
                if model.stopTraining: continue
                model.lr_scheduler.step(torch.tensor(model.epoch_validation_loss).mean())
                model.epoch_reset()
                if "stop" in read_txt(self.run_status_file).lower(): stopTraining = True
                if print_status:
                    model.epoch_status(epoch, self.epochs, epoch_st)
                    model.save()
                        
            if stopTraining:
                INFO("Stopping the training in %d epochs"%(epoch))
                break
        
        # Finally save models
        model_paths = list()
        for model in self.autoencoder_models:
            model.save_final()
            model_paths.append(model.save_path)    
            
        return model_paths