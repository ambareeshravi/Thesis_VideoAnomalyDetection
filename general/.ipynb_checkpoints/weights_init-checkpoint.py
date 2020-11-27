from torch import nn

def weights_init(m, initializer_type = "kaiming_uniform"):
    '''
    Initializes the weigts of the network
    '''
    initializer_type = initializer_type.lower()
    if "kaiming" in initializer_type:
        if "uniform" in initializer_type:
            init_fn = nn.init.kaiming_uniform_
        elif "normal" in initializer_type:
            init_fn = nn.init.kaiming_normal_
    if "xavier"	in initializer_type:
        if "uniform" in initializer_type:
            init_fn = nn.init.xavier_uniform_
        elif "normal" in initializer_type:
            init_fn = nn.init.xavier_normal_
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
        init_fn(m.weight.data)
#     if m.bias:
#         init_fn(m.bias)
#         init_fn(m.bias.data)
#     if isinstance(m, nn.BatchNorm2d):
#         init_fn(m.weight.data, 1.0, 0.02)
#         init_fn(m.bias.data, 0)