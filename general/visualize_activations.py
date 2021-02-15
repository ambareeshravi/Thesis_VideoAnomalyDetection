import sys
sys.path.append("..")

from general import *

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