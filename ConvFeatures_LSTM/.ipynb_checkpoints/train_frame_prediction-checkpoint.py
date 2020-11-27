import sys
sys.path.append("..")
from general import *

from frame_prediction import FrameFeaturePredictor

class FramePredictor_Trainer:
    def __init__(self, 
                 isTrain = True,
                 input_dim = 4096,
                 hidden_dim = 4096,
                 n_frames = 16,
                 useGPU = True,
                ):
        self.device = torch.device('cpu')
        if useGPU and torch.cuda.is_available(): self.device = torch.device("cuda:0")
        self.isTrain = isTrain
        if self.isTrain:
            self.model = FrameFeaturePredictor(input_dim, hidden_dim, self.isTrain, useGPU = useGPU)
            self.model.to(self.device)
        self.n_frames = n_frames
        
    def train(self,
              model_path,
              train_loader,
              val_loader,
              epochs = 100,
              learning_rate = 1e-3,
              batch_size = 32
             ):
        loss_criterion = nn.MSELoss(reduction = "sum")
        optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate, weight_decay = 1e-4)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.75, patience = 4, threshold = 1e-4)
                
        for epoch in range(1, epochs+1):
            epoch_train_loss, epoch_val_loss = list(), list()
            self.model.train()
            for train_batch_idx, (train_features, train_labels) in enumerate(train_loader):
                train_features = train_features.transpose(0,1).to(self.device)
                optimizer.zero_grad()
                train_outputs, train_states = self.model(train_features, self.model.zero_state(self.n_frames))
                loss = loss_criterion(train_features[1:], train_outputs[:-1])
                loss.backward()
                optimizer.step()
                epoch_train_loss.append(loss.item())
            self.model.eval()
            with torch.no_grad():
                for val_batch_idx, (val_features, val_lables) in enumerate(val_loader):
                    val_features = val_features.transpose(0,1).to(self.device)
                    val_outputs = self.model.unroll(val_features[0:1], self.n_frames - 1)
                    val_loss = loss_criterion(val_features[1:], val_outputs[:-1])
                    epoch_val_loss.append(val_loss.item())
                    
            epoch_train_loss = np.mean(epoch_train_loss)
            epoch_val_loss = np.mean(epoch_val_loss)
            lr_scheduler.step(epoch_val_loss)
            print("[%d/%d] Train Loss: %0.4f | Val Loss: %0.4f"%(epoch, epochs+1, epoch_train_loss, epoch_val_loss))
        save_model(self.model, model_path)
        try:
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass
        
    def loss_criterion(self, original, reconstruction):
        return [torch.norm(to_cpu(x) - to_cpu(y)).item() for x,y in zip(original, reconstruction)]
        
    def test_conv_features_lstm(self,
                                model,
                                feat_ext,
                                test_data,
                                batch_size = 8,
                                stackFrames = 16,
                                input_steps = 8,
                                save_as = False):
        model.to(self.device)
        overall_targets, overall_losses = list(), list()
        overall_roc_auc, overall_regularity_scores = list(), list()
        features = list()
        for directory_inputs, directory_labels in tqdm(test_data):
            directory_targets, directory_loss = list(), list()

            directory_input_features = list()
            for idx in range(0, len(directory_inputs), batch_size):
                extracted_features = feat_ext.extract_features(torch.stack(directory_inputs[idx: (idx + batch_size)]))
                directory_input_features += extracted_features
            directory_input_features = torch.stack(directory_input_features)

            for start_idx in range(0, (len(directory_input_features)//stackFrames)*stackFrames, stackFrames):
                test_inputs = directory_input_features[start_idx : (start_idx + stackFrames)] # 16, 1, 128, 128
                test_labels = directory_labels[start_idx : (start_idx + stackFrames)]
                test_inputs = test_inputs.unsqueeze(dim = 1).to(self.device)
                outputs = model.unroll(test_inputs[:input_steps], future_steps = (stackFrames - input_steps))
                loss = self.loss_criterion(test_inputs[1:], outputs[:-1])

                directory_loss += loss
                directory_targets += test_labels[1:]

            regularity_scores = loss_to_regularity(directory_loss)
            try:
                directory_roc_auc = roc_auc_score(directory_targets, regularity_scores)
            except:
                directory_roc_auc = 1.0
            overall_roc_auc.append(directory_roc_auc)
            overall_regularity_scores.append(regularity_scores)

            overall_targets.append(directory_targets)
            overall_losses.append(directory_loss)
    #             overall_encodings.append(directory_encodings)
        overall_targets = np.array(overall_targets)
        overall_losses = np.array(overall_losses)
    #     overall_encodings = np.array(overall_encodings)

        mean_roc_auc = np.mean(overall_roc_auc)

        self.results = {
            "targets": overall_targets,
            "losses": overall_losses,
            "regularity": overall_regularity_scores,
            "AUC_ROC_score": overall_roc_auc,
            "final_AUC_ROC":mean_roc_auc,
        }

        if save_as:
            with open(save_as, "wb") as f:
                pkl.dump(self.results, f)

        return mean_roc_auc