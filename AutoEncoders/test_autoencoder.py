import sys
sys.path.append("..")
from general import *
from general.visualization import *
from general.results_handler import *

from sklearn.svm import OneClassSVM
from sklearn.metrics import *
from scipy.signal import savgol_filter

from pprint import pprint

def pr_auc(y_true, y_pred):
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred, pos_label = ABNORMAL_LABEL)
        return auc(recall, precision)
    except:
        return 0.0

class AE_PredictFunctions:
    def __init__(self,):
        self.patchwise = False
        if "patch" in self.model_file.lower():
            self.patchwise = True
            self.predict = self.predict_patchwise
        self.stacked = False
        if "stack" in self.model_file.lower():
            self.stacked = True
            if self.stackFrames == 1 or self.stackFrames > 16: self.stackFrames = 16
            self.isVideo = True
            self.predict = self.predict_stacked
        if "translat" in self.model_file.lower():
            self.predict = self.predict_translative
        if "attention" in self.model_file.lower():
            self.predict = self.predict_attention
        if "c3d" in self.model_file.lower() or "lstm" in self.model_file.lower() or "rnn" in self.model_file.lower():
            self.isVideo = True
            if self.stackFrames == 1 or self.stackFrames > 16: self.stackFrames = 16
        if "clstm" in self.model_file.lower() and "future" in self.model_file.lower():
            self.predict = self.predict_future
        if "clstm" in self.model_file.lower() and "seq2seq" in self.model_file.lower():
            self.predict = self.predict_sequence
        
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
        stacked_images = inputs.flatten(start_dim=0, end_dim=1)
        with torch.no_grad():
            reconstructions, encodings = self.model(stacked_images.to(self.device))
            reconstructions = reconstructions.unsqueeze(dim = -4)
            print("reconstructions.shape", reconstructions.shape)
            return (reconstructions, encodings)
        
    def predict_translative(self, inputs):
        with torch.no_grad():
            main_encodings = self.model.encoder(inputs)
            r,e = self.et_model(main_encodings)
            main_reconstructions = self.model.decoder(r)
            return main_reconstructions, main_encodings
        
    def predict_attention(self, inputs):
        with torch.no_grad():
            if self.patchwise: inputs = get_patches(inputs, 64, 64)
            if self.stacked: inputs = inputs.flatten(start_dim=0, end_dim=1)
            reconstructions, encodings, attention_activations = self.model(inputs.to(self.device))
            if self.stacked: reconstructions = reconstructions.unsqueeze(dim = -4)
            if self.patchwise: reconstructions = merge_patches(reconstructions)
            return reconstructions, encodings
        
    def predict_future(self, inputs, total_ts = 16):
        with torch.no_grad():
            future_reconstructions, future_encodings = self.model(inputs[:,:,:self.n_seed,:,:], future_steps = (total_ts - self.n_seed))
            seed_reconstructions, seed_encodings = list(), list()
            for s in range(self.n_seed):
                sr, se = self.model(inputs[:,:,s:(s+1),:,:])
                seed_reconstructions += [sr]
                seed_encodings += [se]

            seed_reconstructions = torch.cat(seed_reconstructions, dim = 2)
            seed_encodings = torch.cat(seed_encodings, dim = 2)

            reconstructions = torch.cat((seed_reconstructions, future_reconstructions[:,:,self.n_seed:,:,:]), dim = 2)
            encodings = torch.cat((seed_encodings, future_encodings[:,:,self.n_seed:,:,:]), dim = 2)
            return reconstructions, encodings
        
    def predict_sequence(self, inputs):
        with torch.no_grad():
            predictions, encodings = self.model(inputs, future_steps = self.n_future_steps)
            return predictions, encodings
        
class ReconstructionsMetrics:
    def __init__(self, ):
        self.ssim = SSIM(data_range = 1.0, nonnegative_ssim=True)
        self.psnr = PSNR_LOSS(limit = 1)

    def abs_loss(self, original, reconstruction):
        '''
        should actually be:
        
        torch.sum(torch.abs(a-b), dim = 0)
        '''
        # return [tensor_to_numpy(torch.norm((o-r), dim = 0)) for o,r in zip(original, reconstruction)]
        return [tensor_to_numpy(torch.sum(torch.abs(o-r), dim = 0)) for o,r in zip(original, reconstruction)]
    
    def sqr_loss(self, original, reconstruction):
        '''
        Similar to torch.sum((a-b)**2, dim = 0)
        '''
        # return [tensor_to_numpy(torch.norm((o-r), dim = 0)**2) for o,r in zip(original, reconstruction)]
        return [tensor_to_numpy(torch.sum((o-r)**2, dim = 0)) for o,r in zip(original, reconstruction)]
    
    def ssim_transform(self, x):
        '''
        4 dims, 3 channels
        '''
        if len(x.shape) < 4: x = x.unsqueeze(dim = 0)
        if x.shape[1] != 3: x = x.repeat(1,3,1,1)
        return x
    
    def ssim_loss(self, original, reconstruction):
        '''
        SSIM needs 4 dims for images 1,c,w,h
        '''
        return [float(1-self.ssim(self.ssim_transform(o), self.ssim_transform(r))) for o,r in zip(original, reconstruction)]
    
    def psnr_loss(self, original, reconstruction):
        '''
        0 -> similar | 1 -> different
        '''
        return [float(self.psnr(o, r)) for o,r in zip(original, reconstruction)]
    
class AutoEncoder_Tester(AE_PredictFunctions, ReconstructionsMetrics):
    def __init__(
        self,
        model,
        dataset,
        model_file,
        stackFrames = 64,
        save_vis = False,
        n_seed = 2,
        n_future_steps = 4,
        calcOC_SVM = False,
        applyFilter = False,
        filterWindow = [9,1],
        debug = False,
        recordResults = True,
        useGPU = True
    ):
        self.model = model
        self.dataset = dataset
        self.model_file = model_file
        self.stackFrames = stackFrames
        self.save_vis = save_vis
        self.n_seed = n_seed
        self.n_future_steps = n_future_steps
        self.calcOC_SVM = calcOC_SVM
        self.applyFilter = applyFilter
        self.filterWindow = filterWindow
        self.metric_names = ["absolute", "squared", "ssim", "psnr"]
        self.recordResults = recordResults
        self.debug = debug
        
        self.device = torch.device("cpu")
        if useGPU and torch.cuda.is_available():
            self.device = torch.device("cuda")
            
        self.model.to(self.device)
        self.model.eval()
        
        self.NORMAL_LABEL = NORMAL_LABEL
        self.ABNORMAL_LABEL = ABNORMAL_LABEL
        
        self.isVideo = False
        
        AE_PredictFunctions.__init__(self)
        ReconstructionsMetrics.__init__(self)
        
        self.save_as = ".pkl".join(self.model_file.split(".pth.tar"))
        self.save_path = os.path.split(self.save_as)[0]
       
    def regularity(self, x, applyFilter = True):
        normalized = 1-normalize_error(x)
        if applyFilter and self.applyFilter:
            window = min(self.filterWindow[0], len(x))
            if window % 2 == 0: window -= 1
            return savgol_filter(normalized, window, self.filterWindow[1])
        return normalized
                            
    def plot_regularity(
        self,
        metrics:list,
        labels:list,
        file_name:str,
    ):
        for (m, l) in zip(metrics,labels):
            plt.plot(list(range(len(m))), m, label = l)
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
        return score
        
    def visualize_results(self, inputs, reconstructions, loss_maps):
        format_image = lambda x: tensor_to_numpy(x).transpose(1,2,0)
        results = [np.hstack(visualize_anomalies(format_image(ip), format_image(re), lm)) for (ip, re, lm) in zip(inputs, reconstructions, loss_maps)]
        return results
    
    def get_visualization_path(self, ):
        results_visulization_path = join_paths([self.save_path, "results/"])
        if not os.path.exists(results_visulization_path):
            os.mkdir(results_visulization_path)
        return results_visulization_path
    
    def get_params_dict(self, ):
        params_dict = {
            "encodings": list(),
            "targets": list(),
        }
        for metric in self.metric_names:
            params_dict[metric] = {
                "loss": list(),
                "regularity": list(),
                "auc_roc": list(),
                "pr_roc": list(),
                "classification_report": {},
                "confusion_matrix": {},
                "eer": {},
                "auc_roc_score": {},
                "pr_roc_score": {},
                "best": {}
            }
        return params_dict
    
    def predict_process(self, test_inputs, test_labels, n_frames):
        # reshape if video
        if self.isVideo: test_inputs = test_inputs.transpose(0,1).unsqueeze(dim = 0) # N,C,T,W,H

        # predict and get outputs
        model_outputs = self.predict(test_inputs.to(self.device))
        reconstructions = model_outputs[0]

        # reshape video to frames again
        if self.isVideo:
            test_inputs = test_inputs.squeeze(dim=0).transpose(0,1) # N,C,W,H
            reconstructions = reconstructions.squeeze(dim=0).transpose(0,1) # N,C,W,H
        
        # Some models may not return encodings
        try: encodings = model_outputs[1]
        except: encodings = None
            
        # if last batch, only use the actual frames in the batch
        if n_frames < self.stackFrames:
            test_inputs = test_inputs[-n_frames:]
            test_labels = test_labels[-n_frames:]
            reconstructions = reconstructions[-n_frames:]
            if encodings != None: encodings = encodings[-n_frames:]
        
        # use only 3 channels - for PatchWise models
        test_inputs = to_cpu(test_inputs[..., :3, :, :])
        reconstructions = to_cpu(reconstructions[..., :3, :, :])
        
        # return everything new or that is changed
        return test_inputs, test_labels, reconstructions, encodings
            
    def test(self, return_results = False):
        if "seq" in self.model_file.lower(): return True # Only for testing
        
        results_visulization_path = self.get_visualization_path()
        
        video_level_params = self.get_params_dict()
        
        # iterating through the dataset video by video
        for video_idx, (video_inputs, video_labels) in tqdm(enumerate(self.dataset)):
            
            try: del frame_level_params
            except: pass
            
            frame_level_params = self.get_params_dict()
            
            # check to save visualization
            if self.save_vis: # or video_idx == 0:
                VIS_frames_abs = list()
                VIS_frames_sqr = list()

            # iterating through the video frame by frame
            for frame_idx in range(0, len(video_labels), self.stackFrames):
            
                # getting inputs as frames and labels correspondingly
                test_inputs = torch.stack(video_inputs[frame_idx:(frame_idx + self.stackFrames)]) # T,C,W,H
                test_labels = video_labels[frame_idx:(frame_idx + self.stackFrames)] # N,1
                
                # correcting the shape of last_batch
                n_frames = len(test_labels)
                if n_frames < self.stackFrames:
                    test_inputs = torch.stack(video_inputs[-self.stackFrames:])
                    test_labels = video_labels[-self.stackFrames:]
                
                # process reconstructions, encodings and reference inputs-labels
                test_inputs, test_labels, reconstructions, encodings = self.predict_process(test_inputs, test_labels, n_frames)
                                    
                # note encodings if available
                if self.calcOC_SVM and encodings != None: frame_level_params["encodings"].append(encodings)
                
                # calculate pixel-level loss as 2D maps
                pixel_abs_loss = self.abs_loss(test_inputs, reconstructions) # N,W,H
                pixel_sqr_loss = self.sqr_loss(test_inputs, reconstructions) # N,W,H
                
                frame_level_params["targets"] += test_labels
                # calculate frame-level loss as scalar value
                frame_level_params["absolute"]["loss"] += [pal.sum() for pal in pixel_abs_loss] # N,1 
                frame_level_params["squared"]["loss"] += [psl.sum() for psl in pixel_sqr_loss] # N,1
                frame_level_params["ssim"]["loss"] += self.ssim_loss(test_inputs, reconstructions) # N,1
                frame_level_params["psnr"]["loss"] += self.psnr_loss(test_inputs, reconstructions) # N,1
                
                # get visualization results
                if self.save_vis: # or video_idx == 0:
                     # normalized regularity masks - 0/black normal 1/white abnormal
                    pixel_regularity_abs_mask = [np.abs(1-self.regularity(pal, applyFilter = False)) for pal in pixel_abs_loss] # Normalized between [0,1]
                    pixel_regularity_sqr_mask = [np.abs(1-self.regularity(psl, applyFilter = False)) for psl in pixel_sqr_loss] # Normalized between [0,1]
                    try:                
                        VIS_frames_abs += self.visualize_results(test_inputs, reconstructions, pixel_regularity_abs_mask)
                        VIS_frames_sqr += self.visualize_results(test_inputs, reconstructions, pixel_regularity_sqr_mask)
                    except Exception as e:
                        if self.debug: print("Visualization:", e)
                        self.save_vis = False
            
            video_level_params["targets"].append(np.asarray(frame_level_params["targets"]))

            # Per video calculations
            targets = frame_level_params["targets"]
            plot_regularity_metrics = list()
            for metric in self.metric_names:
                losses = np.asarray(frame_level_params[metric]["loss"])
                regularity = self.regularity(losses)
                video_level_params[metric]["loss"].append(losses)
                video_level_params[metric]["regularity"].append(regularity)
                try: video_level_params[metric]["pr_roc"].append(pr_auc(targets, regularity))
                except Exception as e:
                    if self.debug: print("PR ROC", e)
                    else: pass
                try: video_level_params[metric]["auc_roc"].append(roc_auc_score(targets, regularity))
                except Exception as e:
                    if self.debug: print("AUC ROC", e)
                    else: pass
                plot_regularity_metrics.append(regularity)
                            
            # Visualizations
            self.plot_regularity(
                metrics = [targets] + plot_regularity_metrics,
                labels = ["targets"] + self.metric_names,
                file_name = join_paths([results_visulization_path, "%03d_metrics.png"%(video_idx + 1)])
            )
            
            if self.save_vis: # or video_idx == 0:            
                try:
                    frames_to_video(VIS_frames_abs, join_paths([results_visulization_path, "%03d_AV_abs"%(video_idx + 1)]))
                    frames_to_video(VIS_frames_sqr, join_paths([results_visulization_path, "%03d_AV_sqr"%(video_idx + 1)]))
                except Exception as e:
                    if self.debug: print("Video Generation:", e)
                    self.save_vis = False
                    
            if self.calcOC_SVM: 
                try: video_level_params["encodings"].append(frame_level_params["encodings"])
                except Exception as e:
                    if self.debug: print("Frames Encodings:", e)
                    else: pass
        
        # Convert everything to arrays
        video_level_params["targets"] = np.asarray(video_level_params["targets"]) # V,F,1
        for metric in self.metric_names:
            for key in video_level_params[metric].keys():
                if isinstance(video_level_params[metric][key], list):
                    video_level_params[metric][key] = np.asarray(video_level_params[metric][key]) # V,F,1
        if self.calcOC_SVM: video_level_params["encodings"] = np.asarray(video_level_params["encodings"]) # V,F,E
        
        # Flatten params for overall calculations
        video_level_params["flat_targets"] = flatten_2darray(video_level_params["targets"])
        for metric in self.metric_names:
            video_level_params[metric]["flat_loss"] = flatten_2darray(video_level_params[metric]["loss"])
            video_level_params[metric]["flat_regularity"] = self.regularity(video_level_params[metric]["flat_loss"])
            video_level_params[metric]["agg_regularity"] = flatten_2darray(video_level_params[metric]["regularity"])
        
        # One Class Unsupervised SVM
        svm_score = False
        if self.calcOC_SVM: 
            try:
                video_level_params["flat_encodings"] = np.array([e.flatten() for e in flatten_2darray(video_level_params["encodings"])])
                svm_score = self.OC_SVM(video_level_params["flat_encodings"], video_level_params["flat_targets"], tag = "")
            except Exception as e:
                print("OneCass SVM: Encodings shape: %s, Targets shape: %s, OC_SVM Error: %s:"%(FLT_encodings.shape, FLT_targets.shape, e))
        
        # ---------------- CALCULATING METRICS ------------------- #
        
        for metric in self.metric_names:
            metric_flat_targets = video_level_params["flat_targets"]
            # per video mean metrics
            video_level_params[metric]["auc_roc_score"]["mean"] = np.mean(video_level_params[metric]["auc_roc"])
            video_level_params[metric]["pr_roc_score"]["mean"] = np.mean(video_level_params[metric]["pr_roc"])
            
            # aggregated metrics
            metric_agg_regularity = video_level_params[metric]["agg_regularity"]
            video_level_params[metric]["auc_roc_score"]["agg"] = roc_auc_score(metric_flat_targets, metric_agg_regularity)
            video_level_params[metric]["pr_roc_score"]["agg"] = pr_auc(metric_flat_targets, metric_agg_regularity)
            video_level_params[metric]["eer"]["agg"] = calculate_eer(metric_flat_targets, metric_agg_regularity)
            video_level_params[metric]["classification_report"]["agg"] = classification_report(metric_flat_targets, np.round(metric_agg_regularity), output_dict=True)
            video_level_params[metric]["confusion_matrix"]["agg"] = confusion_matrix(metric_flat_targets, np.round(metric_agg_regularity))
            
            # overall metrics
            metric_overall_regularity = video_level_params[metric]["flat_regularity"]
            video_level_params[metric]["auc_roc_score"]["overall"] = roc_auc_score(metric_flat_targets, metric_overall_regularity)
            video_level_params[metric]["pr_roc_score"]["overall"] = pr_auc(metric_flat_targets, metric_overall_regularity)
            video_level_params[metric]["eer"]["overall"] = calculate_eer(metric_flat_targets, metric_overall_regularity)
            video_level_params[metric]["classification_report"]["overall"] = classification_report(metric_flat_targets, np.round(metric_overall_regularity), output_dict=True)
            video_level_params[metric]["confusion_matrix"]["overall"] = confusion_matrix(metric_flat_targets, np.round(metric_overall_regularity))
            
            # best metrics
            video_level_params[metric]["best"]["threshold"] = thresholdJ(metric_flat_targets, metric_overall_regularity)
            video_level_params[metric]["best"]["predictions"] = scores2labels(metric_overall_regularity, video_level_params[metric]["best"]["threshold"])
            video_level_params[metric]["best"]["classification_report"] = classification_report(metric_flat_targets, video_level_params[metric]["best"]["predictions"], output_dict=True)
            video_level_params[metric]["best"]["confusion_matrix"] = confusion_matrix(metric_flat_targets, video_level_params[metric]["best"]["predictions"])
        
        video_level_params["OC_SVM_Score"] = svm_score
        try:
            video_level_params["encodings"]
            video_level_params["flat_encodings"]
        except:
            pass
        
        # ------- SAVING and DISPLAYING RESUTLS -------- #
        print("="*10, "TEST RESULTS", "="*10)
        pprint(ResultsRecorder.get_results_dict(video_level_params))
        print("="*30)
        
        self.results = video_level_params
        if self.save_as:
            with open(self.save_as, "wb") as f:
                pkl.dump(self.results, f)
        
        if self.recordResults: ResultsRecorder().record_results(self.save_as)
        
        if return_results: return self.results
        return True