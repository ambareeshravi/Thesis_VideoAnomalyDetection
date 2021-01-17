class ModelParser:
    def __init__(self):
        pass
    
    @staticmethod
    def getModelCategory(model_name):
        if "C2D" in model_name: return "C2D"
        elif "C3D" in model_name: return "C3D"
        elif "CLSTM" in model_name: return "CLSTM"
        elif "CRNN" in model_name: return "CRNN"
        else: return False
    
    @staticmethod
    def getModelResolution(model_name):
        if "128" in model_name: return 128
        elif "224" in model_name: return 224
        elif "64" in model_name: return 64
        else: return False
            
    @staticmethod
    def getDatasetType(model_name):
        if "ucsd1" in model_name.lower(): return "UCSD1"
        elif "ucsd2" in model_name.lower(): return "UCSD2"
        elif "subway_entrance" in model_name.lower(): return "SUBWAY_ENTRANCE"
        elif "subway_exit" in model_name.lower(): return "SUBWAY_EXIT"
        elif "avenue" in model_name.lower(): return "AVENUE"
        elif "shagai_tech" in model_name.lower(): return "SHANGAI_TECH"
        elif "street_scene" in model_name.lower(): return "STREET_SCENE"
        else: return False
    
    @staticmethod
    def getImageType(model_name):
        if "normal" in model_name: return "NORMAL"
        elif "gray" in model_name: return "GRAYSCALE"
        elif "flow" in model_name: return "OPTICAL_FLOW"
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
        elif "se" in model_name.lower(): return "SqueezeExcitation"
        elif "pc" in model_name.lower(): return "ParallelConvs"
        elif "wide" in model_name.lower(): return "WideConvs"
        elif "double" in model_name.lower(): return "DoubleHead"
        elif "aac" in model_name.lower(): return "AttentionAugmentedConvs"
        elif "best" in model_name.lower(): return "BestCombo"
        else: return False
        
    @staticmethod
    def getConfig(model_name):
        return {
            "Model": ModelParser.getModelCategory(model_name),
            "Variant": ModelParser.getModelVariant(model_name),
            "Resolution": ModelParser.getModelResolution(model_name),
            "Dataset": ModelParser.getDatasetType(model_name),
            "Image_Type": ModelParser.getImageType(model_name),
            "Loss": ModelParser.getLossType(model_name),
            "Optimizer": ModelParser.getOptimizerType(model_name),
            "DeNoising": ModelParser.isDeNoising(model_name),
            "Attention": ModelParser.isSelfAttentive(model_name)
        }    