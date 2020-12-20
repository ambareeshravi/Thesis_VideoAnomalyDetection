import torch
from torchvision.utils import make_grid, save_image
from PIL import Image
from .utils import *

class ActivationVisualizer:
    def __init__(self,):
        pass
    
    def get_layer_outputs(self, layer, inputs):
        while len(inputs.shape) < 4: 
            inputs = inputs.unsqueeze(dim = 0)
        outputs = list(layer(inputs))
        return outputs[0]
    
    def normalize_tensor(self, x):
        return (x - x.min()) / (x.max() - x.min())
    
    def to_255(self, x):
        if x.max() > 1:
            print(x.max())
            return x
        return (x * 255).type(torch.uint8)
    
    def display(self, x):
        x = x.squeeze()
        x = self.normalize_tensor(x)
        img_array = tensor_to_numpy(x)
        img_array = image_255(img_array)
        img = Image.fromarray(img_array)
        img.show()
        return img
        
    def save_activations(self, input_tensor, file_path, padding = 0):
        if ".png" not in file_path: file_path += ".png"
        input_tensor = input_tensor.squeeze()
        assert len(input_tensor.shape) < 4, "Too many dimensions in the input"
        n = input_tensor.shape[0]
        if len(input_tensor.shape) == 3: input_tensor = input_tensor.unsqueeze(dim = 1)
        grid = make_grid(input_tensor, nrow=int(np.round(np.sqrt(n))), padding = padding)
        save_image(grid, file_path,)