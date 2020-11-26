import sys
sys.path.append("..")

from general import *
from general.model_utils import *

'''
Loss_D - discriminator loss calculated as the sum of losses for the all real and all fake batches (log(D(x))+log(D(G(z)))
).

Loss_G - generator loss calculated as log(D(G(z)))

D(x) - the average output (across the batch) of the discriminator for the all real batch. This should start close to 1 then theoretically converge to 0.5 when G gets better. Think about why this is.

D(G(z)) - average discriminator outputs for the all fake batch. The first number is before D is updated and the second number is after D is updated. These numbers should start near 0 and converge to 0.5 as G gets better. Think about why this is.
'''

class GAN_Trainer:
    def __init__(self, generator, discriminator, floating_labels = True, useGPU = True):
        self.generator = generator
        self.discriminator = discriminator
        self.loss_criterion = nn.BCELoss()
        self.device = torch.device("cpu")
        if useGPU and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            
        self.NORMAL_LABEL = 1.0
        self.ABNORMAL_LABEL = 0.0
        
        self.floating_labels = floating_labels
        
    def get_labels(self, shape, isNormal = True):
        if isNormal:
            labels = torch.ones(*shape) * self.NORMAL_LABEL
        else:
            labels = torch.ones(*shape) * self.ABNORMAL_LABEL
        
        if self.floating_labels:
            return torch.abs(labels - (torch.rand(*labels.shape) * 1e-3)).to(self.device)
        else:
            return labels.to(self.device)
    
    def train(self,
              train_loader,
              val_loader,
              save_path = "./",
              learning_rate = 2e-4,
              epochs = 100,
              monitor_count = 64,
              embedding_cube = 2, 
             ):
        generator_save_path = join_paths([save_path, "generator.pth.tar"])
        discriminator_save_path = join_paths([save_path, "discrimnator.pth.tar"])
        
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr = learning_rate, betas = (0.5, 0.999))
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr = learning_rate, betas = (0.5, 0.999))
        
        epoch_gen_loss = list()
        epoch_dis_loss = list()
        
        self.generator.train()
        self.discriminator.train()
        
        self.fixed_noise = torch.randn(monitor_count, self.generator.embedding_size, embedding_cube, embedding_cube).to(self.device)
        img_list = list()
        
        for epoch in range(1, epochs + 1):
            epoch_st = time()
            batch_gen_loss = list()
            batch_dis_loss = list()
            
            for train_batch_idx, train_batch in enumerate(train_loader):
                self.generator.zero_grad()
                
                train_data, train_labels = train_batch
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                # Train on real batch
                real_data = train_data.to(self.device)
                real_outputs = self.discriminator(real_data).view(-1)
                real_labels = self.get_labels(real_outputs.shape, isNormal = True)
                errD_real = self.loss_criterion(real_outputs, real_labels)
                errD_real.backward()
                D_x = real_outputs.mean().item()
                
                # Train on fake batch
                noisy_embeddings = torch.randn(real_data.shape[0], self.generator.embedding_size, embedding_cube, embedding_cube, device = self.device)
                fake_images = self.generator(noisy_embeddings)
                fake_outputs = self.discriminator(fake_images.detach()).view(-1)
                fake_labels = self.get_labels(fake_outputs.shape)
                errD_fake = self.loss_criterion(fake_outputs, fake_labels)
                errD_fake.backward()
                D_G_z1 = fake_outputs.mean().item()
                errD = errD_real + errD_fake
                discriminator_optimizer.step()
                
                # (2) Update G network: maximize log(D(G(z)))
                self.generator.zero_grad()
                fake_outputs = self.discriminator(fake_images).view(-1)
                fake_labels = self.get_labels(fake_outputs.shape, isNormal = True) # fake images are real for discriminator
                errG = self.loss_criterion(fake_outputs, fake_labels)
                errG.backward()
                D_G_z2 = fake_outputs.mean().item()
                generator_optimizer.step()
                
                batch_gen_loss.append(errG.item())
                batch_dis_loss.append(errD.item())
                
            epoch_gen_loss.append(np.mean(batch_gen_loss))
            epoch_dis_loss.append(np.mean(batch_dis_loss))
            
            print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tTime: %.1f (m)'
                          % (epoch, epochs, np.mean(batch_dis_loss), np.mean(batch_gen_loss), D_x, D_G_z1, D_G_z2, (time()-epoch_st)/60))
                
            if (epoch % 10) == 0:
                with torch.no_grad():
                    fake = self.generator(self.fixed_noise).detach().cpu()
                img_list.append(make_grid(fake, padding=2, normalize=True))
                save_image(make_grid(fake, padding=2, normalize=True), join_paths([save_path, "epoch_%d.png"%(epoch)]))
                
        save_model(self.generator, generator_save_path)
        save_model(self.discriminator, discriminator_save_path)
    
        return generator_save_path, discriminator_save_path
    
    def test(self, stackFrames = 16, save_as = False):
        overall_targets, overall_loss = list(), list()
        overall_roc_auc = list()
        overall_regularity_scores = list()
        for directory_inputs, directory_labels in tqdm(self.dataset):
            directory_targets, directory_loss = list(), list()

            for start_idx in range(0, (len(directory_labels)//stackFrames)*stackFrames, stackFrames):
                test_inputs = torch.stack(directory_inputs[start_idx : (start_idx + stackFrames)]) # 16, 1, 128, 128
                test_labels = directory_labels[start_idx : (start_idx + stackFrames)]
                predictions = self.discriminator(test_inputs.to(self.device))
                directory_loss += tensor_to_numpy(predictions).tolist()
                directory_targets += test_labels
            overall_targets.append(directory_targets)
            overall_loss.append(directory_loss)
            regularity_scores = loss_to_regularity(directory_loss)
            try:
                directory_roc_auc = roc_auc_score(directory_targets, regularity_scores)
            except:
                directory_roc_auc = 1.0
            overall_regularity_scores.append(regularity_scores)
            overall_roc_auc.append(directory_roc_auc)

        overall_targets = np.array(overall_targets)
        overall_loss = np.array(overall_loss)
        overall_regularity_scores = np.array(overall_regularity_scores)
        overall_roc_auc = np.array(overall_roc_auc)

        mean_roc_auc = np.mean(overall_roc_auc)

        self.results = {
            "targets": overall_targets,
            "losses": overall_loss,
            "regularity": overall_regularity_scores,
            "AUC_ROC_score": overall_roc_auc,
            "final_AUC_ROC":mean_roc_auc,
        }

        if save_as:
            with open(save_as, "wb") as f:
                pkl.dump(self.results, f)
        return mean_roc_auc