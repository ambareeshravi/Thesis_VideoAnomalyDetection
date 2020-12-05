import sys
sys.path.append("..")
from general import *
from general.visualization import *

from sklearn.svm import OneClassSVM
from sklearn.metrics import *

from pprint import pprint

class AutoEncoder_Tester:
    def __init__(
        self,
        model,
        dataset,
        patchwise = False,
        stacked = False,
        useGPU = True
    ):
        self.model = model
        self.dataset = dataset
        self.device = torch.device("cpu")
        if useGPU and torch.cuda.is_available():
            self.device = torch.device("cuda")
            
        self.model.to(self.device)
        self.model.eval()
        
        self.NORMAL_LABEL = 1.0
        self.ABNORMAL_LABEL = 0.0
        
        if patchwise:
            self.predict = self.predict_patchwise
        if stacked:
            self.predict = self.predict_stacked
       
    def regularity(self, x):
        return (1-normalize_error(x))
        
    def abs_loss(self, original, reconstruction):
        return [tensor_to_numpy(torch.norm((o-r), dim = 0)) for o,r in zip(original, reconstruction)]
    
    def sqr_loss(self, original, reconstruction):
        return [tensor_to_numpy(torch.norm((o-r), dim = 0)**2) for o,r in zip(original, reconstruction)]
    
    def predict(self, inputs):
        with torch.no_grad():
            return self.model(inputs.to(self.device))
        
    def predict_patchwise(self, inputs):
        patches = get_patches(inputs, 64, 64)
        with torch.no_grad():
            reconstructions, encodings = self.model(patches.to(self.device))
            reconstructions = merge_patches(reconstructions)
            return (reconstructions, encodings)
    
    def predict_stacked(self, inputs):
        stacked_images = inputs.squeeze(dim = -4)
        with torch.no_grad():
            reconstructions, encodings = self.model(stacked_images.to(self.device))
            reconstructions = reconstructions.unsqueeze(dim = -4)
            return (reconstructions, encodings)
                    
    def plot_regularity(self, y_true, y_pred, file_name):
        x = np.array(range(len(y_true)))
        plt.plot(x, y_true, label = "Actual")
        plt.plot(x, y_pred, label = "Predicted")
        plt.plot(x, moving_average(y_pred, 2), label = "Smoothened 2")
        plt.plot(x, moving_average(y_pred), label = "Smoothened 3")
        plt.plot(x, moving_average(y_pred, 5), label = "Smoothened 5")
        plt.xlabel("Time")
        plt.ylabel("Regularity / Normalcy")
        plt.legend()
        plt.savefig(file_name)
        plt.clf()
        try: plt.close()
        except: pass
        
    def OC_SVM(self, features, y_true, tag = ""):
        assert len(features) == len(y_true), "OC_SVM: Number of features and targets are different"
        oc_svm = OneClassSVM()
        y_pred = oc_svm.fit_predict(features)
        score = roc_auc_score(y_true, y_pred)
        print("AUC-ROC Score of %s OneClassSVM: %s"%(tag, score))
        return score
        
    def visualize_results(self, inputs, reconstructions, loss_maps):
        format_image = lambda x: tensor_to_numpy(x).transpose(1,2,0)
        results = [np.hstack(visualize_anomalies(format_image(ip), format_image(re), lm)) for (ip, re, lm) in zip(inputs, reconstructions, loss_maps)]
        return results
        
    def test(self, stackFrames = 1, isVideo = False, save_as = "./", save_vis = True):
        save_path = os.path.split(save_as)[0]
        results_visulization_path = join_paths([save_path, "results/"])
        if not os.path.exists(results_visulization_path):
            os.mkdir(results_visulization_path)
        
        VL_targets = list()
        VL_abs_losses = list()
        VL_sqr_losses = list()
        VL_abs_regularity_scores = list()
        VL_sqr_regularity_scores = list()
        VL_abs_rocauc_scores = list()
        VL_sqr_rocauc_scores = list()
        VL_encodings = list()
        
        # iterating through the dataset video by video
        for video_idx, (video_inputs, video_labels) in tqdm(enumerate(self.dataset)):
            
            FL_targets = list()
            FL_abs_losses = list()
            FL_sqr_losses = list()
            FL_encodings = list()
            
            # check to save visualization
            if save_vis or video_idx == 0:
                VIS_frames_abs = list()
                VIS_frames_sqr = list()

            # iterating through the video frame by frame
            for frame_idx in range(0, len(video_labels), stackFrames):
                # getting inputs as frames and labels correspondingly
                test_inputs = torch.stack(video_inputs[frame_idx:(frame_idx + stackFrames)]) # T,C,W,H
                test_labels = video_labels[frame_idx:(frame_idx + stackFrames)] # N,1
                
                # correcting the shape of last_batch
                n_frames = len(test_labels)
                if n_frames < stackFrames:
                    test_inputs = torch.stack(video_inputs[-stackFrames:])
                    test_labels = video_labels[-stackFrames:]
                
                # reshape if video
                if isVideo:
                    test_inputs = test_inputs.transpose(0,1).unsqueeze(dim = 0) # N,C,T,W,H
                    
                # predict and get outputs
                model_outputs = self.predict(test_inputs.to(self.device))
                reconstructions = model_outputs[0]
                
                # reshape video to frames again
                if isVideo:
                    test_inputs = test_inputs.squeeze(dim=0).transpose(0,1) # N,C,W,H
                    reconstructions = reconstructions.squeeze(dim=0).transpose(0,1) # N,C,W,H
                
                # if last batch, only use the actual frames in the batch
                if n_frames < stackFrames:
                    test_inputs = test_inputs[-n_frames:]
                    test_labels = test_labels[-n_frames:]
                    reconstructions = reconstructions[-n_frames:]
                    
                # note encodings
                try:
                    output_encodings = tensor_to_numpy(model_outputs[1])
                    if n_frames < stackFrames:
                        output_encodings = output_encodings[-n_frames:]
                    FL_encodings.append(output_encodings)
                except Exception as e:
                    print("Frames Encodings:", e)
                
                # use only 3 channels - for PatchWise models
                test_inputs = to_cpu(test_inputs[..., :3, :, :])
                reconstructions = to_cpu(reconstructions[..., :3, :, :])
                
                # calculate pixel-level loss as 2D maps
                pixel_abs_loss = self.abs_loss(test_inputs, reconstructions) # N,W,H
                pixel_sqr_loss = self.sqr_loss(test_inputs, reconstructions) # N,W,H
                
                # calculate frame-level loss as scalar value
                frame_abs_loss = [pal.sum() for pal in pixel_abs_loss] # N,1
                frame_sqr_loss = [psl.sum() for psl in pixel_sqr_loss] # N,1
                
                # normalized regularity masks - 0/black normal 1/white abnormal
                pixel_regularity_abs_mask = [np.abs(1-self.regularity(pal)) for pal in pixel_abs_loss] # Normalized between [0,1]
                pixel_regularity_sqr_mask = [np.abs(1-self.regularity(psl)) for psl in pixel_sqr_loss] # Normalized between [0,1]
                
                FL_targets += test_labels
                
                FL_abs_losses += frame_abs_loss
                FL_sqr_losses += frame_sqr_loss
                
                # get visualization results
                if save_vis or video_idx == 0:
                    try:                
                        VIS_frames_abs += self.visualize_results(test_inputs, reconstructions, pixel_regularity_abs_mask)
                        VIS_frames_sqr += self.visualize_results(test_inputs, reconstructions, pixel_regularity_sqr_mask)
                    except Exception as e:
                        print("Visualization:", e)
                
            VL_targets.append(np.asarray(FL_targets))
            VL_abs_losses.append(np.asarray(FL_abs_losses))
            VL_sqr_losses.append(np.asarray(FL_sqr_losses))
            VL_abs_regularity_scores.append(self.regularity(np.array(FL_abs_losses)))
            VL_sqr_regularity_scores.append(self.regularity(np.array(FL_sqr_losses)))
            
            # Calculate AUC-ROC scores per video
            try:
                VL_abs_rocauc_scores.append(roc_auc_score(FL_targets, VL_abs_regularity_scores[-1]))
                VL_sqr_rocauc_scores.append(roc_auc_score(FL_targets, VL_sqr_regularity_scores[-1]))
            except Exception as e:
                print("Calculation of AUC-ROC score", e)
                
            # Visualizations
            self.plot_regularity(FL_targets, VL_abs_regularity_scores[-1], join_paths([results_visulization_path, "%03d_R_abs.png"%(video_idx + 1)]))
            self.plot_regularity(FL_targets, VL_sqr_regularity_scores[-1], join_paths([results_visulization_path, "%03d_R_sqr.png"%(video_idx + 1)]))
            
            if save_vis or video_idx == 0:            
                try:
                    frames_to_video(VIS_frames_abs, join_paths([results_visulization_path, "%03d_AV_abs"%(video_idx + 1)]))
                    frames_to_video(VIS_frames_sqr, join_paths([results_visulization_path, "%03d_AV_sqr"%(video_idx + 1)]))
                except Exception as e:
                    print("Video Generation:", e)
            
            try:
                VL_encodings.append(FL_encodings)
            except Exception as e:
                print("Frames Encodings:", e)
        
        VL_targets = np.asarray(VL_targets) # V,F,1
        VL_abs_losses = np.asarray(VL_abs_losses) # V,F,1
        VL_sqr_losses = np.asarray(VL_sqr_losses) # V,F,1
        VL_abs_regularity_scores = np.asarray(VL_abs_regularity_scores) # V,F,1
        VL_sqr_regularity_scores = np.asarray(VL_sqr_regularity_scores) # V,F,1
        VL_abs_rocauc_scores = np.asarray(VL_abs_rocauc_scores) # V,1
        VL_sqr_rocauc_scores = np.asarray(VL_sqr_rocauc_scores) # V,1
        VL_encodings = np.asarray(VL_encodings) # V,F,E
        
        FLT_targets = flatten_2darray(VL_targets)
            
        # Regularize everything together
        FLT_abs_regularity = self.regularity(flatten_2darray(VL_abs_losses))
        FLT_sqr_regularity = self.regularity(flatten_2darray(VL_sqr_losses))
        
        # Aggregate regularized per video scores
        FLT_agg_abs_regularity = flatten_2darray(VL_abs_regularity_scores)
        FLT_agg_sqr_regularity = flatten_2darray(VL_sqr_regularity_scores)
        
        # One Class Unsupervised SVM
        try:
            FLT_encodings = np.array([e.flatten() for e in flatten_2darray(VL_encodings)])
            svm_score = 0.0
            svm_score = self.OC_SVM(FLT_encodings, FLT_targets, tag = "")
        except Exception as e:
            print("OneCass SVM: Encodings shape: %s, Targets shape: %s, OC_SVM Error: %s:"%(FLT_encodings.shape, FLT_targets.shape, e))
            
        # Mean video roc-auc
        mean_abs_vid_aucroc = VL_abs_rocauc_scores.mean()
        mean_sqr_vid_aucroc = VL_sqr_rocauc_scores.mean()
        
        # aggregated regularity roc-auc
        # video-wise regularized scores [CORRECT and INTUITIVE]
        agg_abs_reg_aucroc = roc_auc_score(FLT_targets, FLT_agg_abs_regularity)
        agg_abs_reg_report = classification_report(FLT_targets, np.round(FLT_agg_abs_regularity))
        agg_abs_conf_matrix = confusion_matrix(FLT_targets, np.round(FLT_agg_abs_regularity))
        agg_abs_eer = calculate_eer(FLT_targets, FLT_agg_abs_regularity)
        
        agg_sqr_reg_aucroc = roc_auc_score(FLT_targets, FLT_agg_sqr_regularity)
        agg_sqr_reg_report = classification_report(FLT_targets, np.round(FLT_agg_sqr_regularity))
        agg_sqr_conf_matrix = confusion_matrix(FLT_targets, np.round(FLT_agg_sqr_regularity))
        agg_sqr_eer = calculate_eer(FLT_targets, FLT_agg_sqr_regularity)
        
        # overall roc-auc
        overall_abs_aucroc = roc_auc_score(FLT_targets, FLT_abs_regularity)
        overall_abs_eer = calculate_eer(FLT_targets, FLT_abs_regularity)
        overall_abs_report = classification_report(FLT_targets, np.round(FLT_abs_regularity))
        
        overall_sqr_aucroc = roc_auc_score(FLT_targets, FLT_sqr_regularity)
        overall_sqr_eer = calculate_eer(FLT_targets, FLT_sqr_regularity)
        overall_sqr_report = classification_report(FLT_targets, np.round(FLT_sqr_regularity))
        
        self.results = {
            "AUC_ROC": {
                "mean_abs_vid_aucroc": mean_abs_vid_aucroc,
                "mean_sqr_vid_aucroc": mean_sqr_vid_aucroc,
                "agg_abs_reg_aucroc": agg_abs_reg_aucroc,
                "agg_sqr_reg_aucroc": agg_sqr_reg_aucroc,
                "agg_abs_eer": agg_abs_eer,
                "agg_sqr_eer": agg_sqr_eer,
                "overall_abs_aucroc": overall_abs_aucroc,
                "overall_sqr_aucroc": overall_sqr_aucroc,
                "overall_abs_eer": overall_abs_eer,
                "overall_sqr_eer": overall_sqr_eer
            },
            "classification_reports": {
                "agg_abs_reg_report": agg_abs_reg_report,
                "agg_sqr_reg_report": agg_sqr_reg_report,
                "overall_abs_report": overall_abs_report,
                "overall_sqr_report": overall_sqr_report
            },
            "confusion_matrices": {
                "agg_abs_conf_matrix": agg_abs_conf_matrix,
                "agg_sqr_conf_matrix": agg_sqr_conf_matrix,
            },
            "params": {
                "targets": VL_targets,
                "encodings": VL_encodings,
                "abs_losses": VL_abs_losses,
                "sqr_losses": VL_sqr_losses,
                "abs_regularity": VL_abs_regularity_scores,
                "sqr_regularity": VL_sqr_regularity_scores,
                "abs_rocauc": VL_abs_rocauc_scores,
                "sqr_rocauc": VL_sqr_rocauc_scores
            },
            "OC_SVM_Score": svm_score
        }
        
        print("-"*20, "TEST RESULTS", "-"*20)
        pprint(self.results["AUC_ROC"])
        print("="*54)
                    
        if save_as:
            with open(save_as, "wb") as f:
                pkl.dump(self.results, f)