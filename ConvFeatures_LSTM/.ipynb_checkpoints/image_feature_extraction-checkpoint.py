import sys
sys.path.append("..")
from general import *

from torchvision import models as torchvision_models

class ImageFeatureExtractor:
    def __init__(self, model_type = "vgg16", useGPU = True):
        self.device = torch.device("cpu")
        if useGPU and torch.cuda.is_available(): self.device = torch.device("cuda:0")
        self.model_type = model_type.lower()
        if "squeeze" in self.model_type:
            self.feature_extractor = torchvision_models.squeezenet1_1(pretrained=True, progress=False)
        elif "vgg16" in self.model_type:
            vgg = torchvision_models.vgg16_bn(pretrained=True, progress=False)
            self.feature_extractor = nn.Sequential(
                                            vgg.features,
                                            vgg.avgpool,
                                            nn.Flatten(),
                                            vgg.classifier[:4]
                                        )
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
    
    def color_to_gray(self, x):
        new_shape = [1]*len(x.shape)
        if x.shape[-3] == 1:
            new_shape[-3] = 3
            return x.repeat(*new_shape)
        else:
            return x
        
    def extract_features(self, images):
        images = self.color_to_gray(images)
        with torch.no_grad():
            return self.feature_extractor(images.to(self.device)).detach().cpu()