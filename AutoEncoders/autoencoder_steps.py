import sys
sys.path.append("..")
from general.all_imports import *
from general.utils import *
from SVM.deep_SVDD import DeepSVDD

class AutoEncoderHelper:
    def __init__(
        self,
        model_file,
        noise_var = 0.1
    ):
        self.model_file = model_file
        
        # Variational Step
        if "vae" in self.model_file.lower():
            self.step = self.vae_step
        
        # DeNoising Step
        self.addNoise = False
        if "nois" in self.model_file.lower():
            self.addNoise = True
            self.noise_var = noise_var
        
        # PatchWise Step
        self.patchwise = False
        if "patch" in self.model_file.lower():
            self.patchwise = True
            self.step = self.patch_step
        
        # Stacked step
        self.stacked = False
        if "stack" in self.model_file.lower():
            self.stacked = True
            self.step = self.stack_step
        
        # Noose step
        if "noose" in self.model_file.lower():
            self.noose_factor = 1.0
            self.step = self.noose_step
        
        # Translation step
        self.isET = False
        if "translat" in self.model_file.lower():
            self.isET = True
            self.step = self.double_translative_step
            from AutoEncoders.embedding_translation import EmbeddingTranslator
            self.et_model = EmbeddingTranslator()
            self.et_criterion = nn.MSELoss()
            self.et_optimizer = torch.optim.Adam(self.et_model.parameters(), lr = 1e-4, weight_decay = 1e-5)
            self.et_model.to(self.device)
            
        # Origin Push step
        if "origin_push" in self.model_file.lower():
            self.step = self.origin_push
        
        # Gaussian Push step
        self.is_gaussian = False
        if "gaussian" in self.model_file.lower():
            self.is_gaussian = True
            self.step = self.gaussian_push
            self.is_zero_push = False
            
            # Gaussian Zero push
            if "zero" in self.model_file.lower():
                self.is_zero_push = True
            self.sigma = 0.1
        
        # Attention step
        if "softmax_attention" in self.model_file.lower():
            self.isAttention = True
            self.step = self.softmax_attention_step
            
        if "conv_attention" in self.model_file.lower():
            self.isAttention = True
            self.step = self.attention_step
        
        # Future prediction step
        if "clstm" in self.model_file.lower() and "future" in self.model_file.lower():
            self.step = self.future_step
            
        if "clstm" in self.model_file.lower() and "seq2seq" in self.model_file.lower():
            self.step = self.sequence_step
            
        if "rnn" in self.model_file.lower() or "lstm" in self.model_file.lower() or "gru" in self.model_file.lower():
            self.step = self.masked_step
        
        self.isSVDD_enabled = False
        if "svdd" in self.model_file.lower():
            self.isSVDD_enabled = True
            svdd_type = "soft_boundary"
            if "one" in self.model_file.lower():
                svdd_type = "one_class"
            self.svdd = DeepSVDD(objective = svdd_type)
            self.svdd_init = False
            self.svdd_warmup_count = 0
            
        if "alw" in self.model_file.lower():
            self.step = self.alw_step
        
        self.isGaussianBlur = False
        if "blur" in self.model_file.lower():
            self.isGaussianBlur = True
            self.gaussian_blur = transforms.GaussianBlur((3,3))
            
        if "best" in self.model_file.lower():
            self.step = self.best_step
            
    def epoch_reset(self,):
        train_loss = np.mean(self.epoch_train_loss)
        val_loss = np.mean(self.epoch_validation_loss)
        
        self.history["train_loss"].append(train_loss)
        self.history["validation_loss"].append(val_loss)
        
        self.epoch_train_loss = list()
        self.epoch_validation_loss = list()
        
        if self.isSVDD_enabled:
            try:
                self.svdd.history["train_loss"].append(np.mean(self.svdd.history["epoch_train_loss"]))
                self.svdd.history["epoch_train_loss"] = list()
                self.svdd.history["val_loss"].append(np.mean(self.svdd.history["epoch_val_loss"]))
                self.svdd.history["epoch_val_loss"] = list()
                self.svdd.lr_scheduler.step(self.svdd.history["val_loss"][-1])
            except Exception as e:
                print("Problem with SVDD epoch end", e)
                    
    def get_inputs(self, images):
        if self.addNoise:
            return add_noise(images, var = self.noise_var)
        if self.isGaussianBlur:
            return self.gaussian_blur(images)
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
        stacked_images = images.flatten(start_dim = 0, end_dim = 1)
        reconstructions, encodings = self.model(stacked_images)
        return self.loss_criterion(stacked_images, reconstructions)
    
    def vae_step(self, images):
        reconstructions, latent_mu, latent_logvar = self.model(self.get_inputs(images))
        return self.model.vae_loss(images, reconstructions, latent_mu, latent_logvar)
        
    def noose_step(self, images):
        reconstructions, encodings = self.model(self.get_inputs(images))
        return self.loss_criterion(images, reconstructions) + (self.noose_factor * self.loss_criterion(encodings, encodings.mean(dim = 0)))
    
    def origin_push(self, images, lambda_ = 1e-10):
        reconstructions, encodings = self.model(self.get_inputs(images))
        return self.loss_criterion(images, reconstructions) - (lambda_ * torch.sum(encodings))
    
    def gaussian(self, x):
        x = x.flatten(start_dim = 1, end_dim = -1)
        to_push = x.mean(dim=0)
        if self.is_zero_push: to_push = torch.zeros_like(x)
        return torch.exp(-torch.norm((x-to_push), dim = -1)**2 / (2 * self.sigma**2))
    
    def gaussian_push_loss(self, encodings, lambda_ = 1e-4):
        return lambda_ * torch.sum(1 - self.gaussian(encodings))
    
    def gaussian_push(self, images):
        reconstructions, encodings = self.model(self.get_inputs(images))
        return self.loss_criterion(images, reconstructions) + self.gaussian_push_loss(encodings)
    
    def double_translative_step(self, images):
        reconstructions, encodings = self.model(self.get_inputs(images))
        
        try:
            self.et_model.zero_grad()
            d_encodings = encodings.detach()
            r, e = self.et_model(d_encodings)
            et_loss = self.et_criterion(d_encodings, r)
            et_loss.backward()
            self.et_optimizer.step()
        except:
            pass
        
        return self.loss_criterion(images, reconstructions)
    
    def softmax_attention_step(self, images):
        reconstructions, encodings, attention_activations = self.model(self.get_inputs(images))
        loss = self.loss_criterion(images, reconstructions)
        return loss
    
    def attention_step(self, images):
        if self.patchwise: images = get_patches(images)
        if self.stacked: images = images.flatten(start_dim = 0, end_dim = 1)
        
        reconstructions, encodings, attention_activations = self.model(self.get_inputs(images))
        loss = self.loss_criterion(images, reconstructions) + self.model.attention_loss(attention_activations)
        if self.is_gaussian: loss += self.gaussian_push_loss(encodings)
        return loss
    
    def alw_step(self, images, lambda_ = 1e-3):
        reconstructions, encodings = self.model(self.get_inputs(images))
        loss = self.loss_criterion(images, reconstructions) + (lambda_ * self.model.get_attention_loss())
        return loss
    
    def best_step(self, images):
        reconstructions, encodings = self.model(self.get_inputs(images))
        loss = self.loss_criterion(images, reconstructions) + self.model.auxillary_loss()
        return loss
    
    def future_step(self, images):
        reconstructions, encodings = self.model(self.get_inputs(images)[:,:,:-1,:,:])
        return self.loss_criterion(images[:,:,1:,:,:], reconstructions)
    
    def masked_step(self, images):
        bs,c,ts,w,h = images.shape
        mask_indices = np.unique(np.random.randint(low = 0, high = ts, size = ts // 3))
        masked_images = images[:,:,mask_indices,:,:] * 0.0
        reconstructions, encodings = self.model(images)
        loss = self.loss_criterion(images[:,:,mask_indices,:,:], reconstructions[:,:,mask_indices,:,:])
        return loss
    
    def sequence_step(self, images):
        video_chunks = self.get_inputs(images) # bs,c,ts,w,h
        bs,c,ts,w,h = video_chunks.shape
        current_steps = np.random.choice(list(range(ts//3, ts-2)))
        future_steps = ts - current_steps
        predictions, encodings = self.model(video_chunks[:,:,:current_steps,:,:], future_steps = future_steps)
        return self.loss_criterion(video_chunks[:,:,current_steps:,:,:], predictions)
            
    def save(self,):
        if self.isSVDD_enabled:
            self.model.objective = self.svdd.objective
            self.model.R = self.svdd.R
            self.model.c = self.svdd.c
            self.model.nu = self.svdd.nu
            
        save_model(self.model, self.model_file)
        if self.isET:
            save_model(self.et_model, "_ET.pth.tar".join(self.model_file.split(".pth.tar")))
    
    def save_final(self,):
        self.save()
        plot_stat(self.history, os.path.split(self.model_file)[-1], self.save_path)
        with open(os.path.join(self.save_path, "train_stats.pkl"), "wb") as f:
            pkl.dump(self.history, f) 