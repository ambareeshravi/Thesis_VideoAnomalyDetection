import sys
sys.path.append("..")

from general import *

def mean_adjust(x_org, threshold = 0.0):
    try: x = x_org.copy()
    except: x = x_org.detach().clone()
    for xi in x:
        xi[np.where(xi > ((1 + threshold) * xi.mean()))] = 0
    return x

def plt_save(x, file_name, cmap = "viridis"):
    if (len(x.shape) > 2) and x.shape[-1] == 1: x = x.squeeze()
    if ".png" not in file_name: file_name += ".png"
    plt.imsave(file_name, x, cmap = cmap)
    
def reordered_numpy(x):
    return tensor_to_numpy(x.permute(1,2,0))

def equalizeHist(x):
    x = image_255(x)
    if len(x.shape) > 2:
        R, G, B = cv2.split(x)
        output1_R = cv2.equalizeHist(R)
        output1_G = cv2.equalizeHist(G)
        output1_B = cv2.equalizeHist(B)
        return cv2.merge((output1_R, output1_G, output1_B)) / 255.
    else:
        return cv2.equalizeHist(x) / 255.
    
def create_grid(x, padding = 0):
    c,w,h = x.shape
    g = make_grid(torch.tensor(x).unsqueeze(dim=1), nrow=round(np.sqrt(c)), padding = padding)
    return g.permute(1,2,0).numpy()

class LayerActivationMaps:
    def __init__(self, ):
        '''
        this stores the activations for each run batch_size x map_channels x map_height x map_width
        
        lam = LayerActivationMaps()
        lam.make_hooks(['act_block1', 'act_block2','act_block3'], [model.act_block1, model.act_block2, model.act_block3])
        '''
        self.activations = dict()
    
    def get_activation(self, name):
        '''
        Creates and registers forward hooks to get activation for every forward pass
        '''
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def make_hooks(self, layer_names, layers):
        '''
        Registers a hook for each of the layers in the network
        '''
        for layer_name, layer in zip(layer_names, layers):
            layer.register_forward_hook(self.get_activation(layer_name))
    
    def get_activations(self):
        '''
        Returns activations for every layer as a dict
        '''
        return self.activations