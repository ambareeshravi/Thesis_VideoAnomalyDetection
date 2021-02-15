import sys
sys.path.append("..")

from general.utils import *

class ModelParser:
    def __init__(self):
        pass
    
    @staticmethod
    def getModelCategory(model_name):
        if "|" in model_name:
            return os.path.split(model_name.split("|")[0])[-1]
        elif "-" in model_name:
            return os.path.split(model_name.split("-")[0])[-1]
        else:
            return os.path.split(model_name.split("_")[0])[-1]
        
        if "C2D" in model_name: return "C2D"
        elif "C3D" in model_name: return "C3D"
        elif "CLSTM" in model_name: return "CLSTM"
        elif "CRNN" in model_name: return "CRNN"
        else: return False
    
    @staticmethod
    def getModelResolution(model_name):
        if "128" in model_name.lower(): return 128
        elif "224" in model_name.lower(): return 224
        elif "64" in model_name.lower(): return 64
        else: return False
            
    @staticmethod
    def getDatasetType(model_name):
        if "ucsd1" in model_name.lower(): return "UCSD1"
        elif "ucsd2" in model_name.lower(): return "UCSD2"
        elif "subway_entrance" in model_name.lower(): return "SUBWAY_ENTRANCE"
        elif "subway_exit" in model_name.lower(): return "SUBWAY_EXIT"
        elif "avenue" in model_name.lower(): return "AVENUE"
        elif "shangai_tech" in model_name.lower(): return "SHANGAI_TECH"
        elif "street_scene" in model_name.lower(): return "STREET_SCENE"
        elif "ham" in model_name.lower(): return "HAM10000"
        elif "distraction" in model_name.lower(): return "IR_DISTRACTION"
        elif ("mv_tec" in model_name.lower()) or ("mv-tec" in model_name.lower()): return "MV_TEC"
        else: return False
    
    @staticmethod
    def getImageType(model_name):
        if "normal" in model_name.lower(): return "NORMAL"
        elif "gray" in model_name.lower(): return "GRAYSCALE"
        elif "flow_mask" in model_name.lower(): return "OPTICAL_FLOW_MASK"
        elif "flow" in model_name.lower(): return "OPTICAL_FLOW"
        else: return False
    
    @staticmethod
    def getLossType(model_name):
        if "mse" in model_name.lower(): return "MSE"
        elif "bce" in model_name.lower(): return "BCE"
        elif "quality" in model_name.lower(): return "QUALITY"
        elif "manifold" in model_name.lower(): return "MANIFOLD"
        elif "weighted" in model_name.lower(): return "WEIGHTED_SIMILARITY"
        elif "psnr" in model_name.lower(): return "PSNR"
        else: return False
    
    @staticmethod
    def isDeNoising(model_name):
        return "denoising" in model_name.lower()
    
    @staticmethod
    def isSelfAttentive(model_name):
        return "attention" in model_name.lower()
    
    @staticmethod
    def getOptimizerType(model_name):
        if "sgd" in model_name.lower(): return "SGD"
        elif "adam" in model_name.lower(): return "ADAM"
        elif "adagrad" in model_name.lower(): return "ADAGRAD"
        else: return False
        
    @staticmethod
    def getModelVariant(model_name):
        if "vae" in model_name.lower(): return "Variational"
        elif "res" in model_name.lower(): return "RESNet"
        elif "acb" in model_name.lower(): return "ACB"
        elif "dp" in model_name.lower(): return "Dropouts"
        elif "sqzex" in model_name.lower(): return "SqueezeExcitation"
        elif "pc" in model_name.lower(): return "ParallelConvs"
        elif "wide" in model_name.lower(): return "WideConvs"
        elif "double" in model_name.lower(): return "DoubleHead"
        elif "aac" in model_name.lower(): return "AttentionAugmentedConvs"
        elif "best" in model_name.lower(): return "BestCombo"
        elif "normal" in model_name.lower(): return "Normal"
        else: return "Normal"
        
    @staticmethod
    def getConfig(model_name):
        return OrderedDict([
            ("Model", ModelParser.getModelCategory(model_name)),
            ("Model_Path", model_name[-40:]),
            ("Variant", ModelParser.getModelVariant(model_name)),
            ("Dataset", ModelParser.getDatasetType(model_name)),
            ("Image_Type", ModelParser.getImageType(model_name)),
            ("Resolution", ModelParser.getModelResolution(model_name)),
            ("Loss", ModelParser.getLossType(model_name)),
            ("Optimizer", ModelParser.getOptimizerType(model_name)),
            ("DeNoising", ModelParser.isDeNoising(model_name)),
            ("Attention", ModelParser.isSelfAttentive(model_name))
        ])
    
class ResultsRecorder:
    def __init__(
        self,
        results_file:str = "VAD_AE_results.csv", 
        sort_by:str = ["Dataset", "Model"]
    ):
        self.results_file = results_file
        self.sort_by = sort_by
            
    @staticmethod
    def get_results_dict(
        results:dict,
        metrics:list = ["absolute", "squared", "ssim", "psnr"],
        scores:list = ["auc_roc_score", "pr_roc_score", "eer"],
        cr_keys:list = ['precision', 'recall', 'f1-score', 'support']
    ):
        results_dict = OrderedDict()
        def insert(k,v): results_dict[k] = v

        for score in scores:
            for metric in metrics:
                if isinstance(results[metric][score], dict):
                    for key, value in results[metric][score].items():
                        insert("%s_%s_%s"%(score, key, metric), value)
                else:
                    insert("%s_%s"%(score, metric), results[metric][score])
                    
        results_dict = OrderedDict(sorted(results_dict.items()))
        
        for type_ in ["agg", "overall"]:
            for cr_key in cr_keys:
                for metric in metrics:
                    insert("%s_%s_%s"%(cr_key, metric, type_), results[metric]["best"][type_]["classification_report"]["weighted avg"][cr_key])

            for metric in metrics:
                insert("conf_mat_%s_%s"%(metric, type_), ",".join(list(map(str, flatten_2darray(results[metric]["best"][type_]["confusion_matrix"]).tolist()))))
            
        return results_dict
    
    def getResultsFile(self, model_name):
        return "%s_results.csv"%(ModelParser.getModelCategory(model_name))
    
    def readCsvPandas(self):
        try: return pd.read_csv(self.results_file, index_col=False)
        except: return pd.DataFrame()
    
    def storeCsvPandas(self, results_df):
        results_df.to_csv(self.results_file, index=False)
    
    def get_record(self, model_results_file):
        results_dict = OrderedDict()
        model_config = ModelParser.getConfig(model_results_file)
        results = load_pickle(model_results_file)
        metrics_dict = ResultsRecorder.get_results_dict(results)
        results_dict.update(model_config)
        results_dict.update(metrics_dict)
        return results_dict
    
    def record_results(self, model_results_file):
        record = self.get_record(model_results_file)
        results_file = self.getResultsFile(model_results_file)
        # read previous results
        results_df = self.readCsvPandas()
        if results_df.empty:
            results_df = pd.DataFrame(columns=list(record.keys()))
        # add new results
        results_df = results_df.append(record, ignore_index=True)
        # sort and remove duplicates
        results_df = results_df.round(4)
        results_df.sort_values(self.sort_by, inplace=True)
        results_df = results_df.drop_duplicates()
        # save results data
        self.storeCsvPandas(results_df)
        return True