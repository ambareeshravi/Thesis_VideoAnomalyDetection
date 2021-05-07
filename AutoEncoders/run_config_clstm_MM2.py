import argparse
import sys
sys.path.append("..")

# from general.utils import INFO

# Import AutoEncoder trainer and tester scripts
from train_autoencoder import *
from test_autoencoder import *

# Import all available models
from ConvLSTM_AE.ConvLSTM_AE import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Parameters to run the training")
    parser.add_argument("--data_path", type = str, help="Path to read data from")
    parser.add_argument("--model_path", type = str, help="Path to store the models")
    args = parser.parse_args()
    
    # Editable
    IMAGE_SIZE = 64
    EPOCHS = 50
    BATCH_SIZE = 64
    IMAGE_TYPE = "grayscale"
    MODEL_PATH = args.model_path
    if not os.path.exists(MODEL_PATH): os.mkdir(MODEL_PATH)
    DATA_PATH = os.path.join(args.data_path, "MOVING_MNIST")
    OPTIMIZER_TYPE = "adam"
    LOSS_TYPE = "mse"
    DENOISING = False
    
    isVideo = True
    stackFrames = 20
    asImages = False
        
    # Manual
    DATA_TYPE = "MovingMNIST" 
    
    # 1. Training
    train_data = MovingMNIST(parent_path = DATA_PATH)
    CHANNELS = 1
    channels = CHANNELS

    train_loader, val_loader = get_data_loaders(train_data, batch_size = BATCH_SIZE)
    INFO("TRAINING DATA READY")
    
    kwargs = {
        "image_size": IMAGE_SIZE,
        "channels": channels,
        "filter_count": [64, 64, 64, 128, 128],
        "filter_sizes": [3, 3, 3, 3, 3], 
        "filter_strides": [2, 2, 2, 2, 2],
        "n_r_layers": 5,
        "disableRecDeConv": False,
    }
    
    COMPLETE_MODELS_LIST = list()
    for model_variant in [CRNN_AE, BiCRNN_AE, CRNN_AE_Seq2Seq, CLSTM_AE, BiCLSTM_AE, CLSTM_AE_Seq2Seq, CGRU_AE, BiCGRU_AE, CGRU_AE_Seq2Seq]:
        for disableRecDecConv in [False, True]:
            for n_r_layers in range(1, len(kwargs["filter_sizes"]) + 1):
                kwargs['n_r_layers'] = n_r_layers
                kwargs['disableRecDeConv'] = disableRecDecConv
                COMPLETE_MODELS_LIST.append(model_variant(**kwargs))
    MODELS_LIST = COMPLETE_MODELS_LIST[75:80] # 25-30, 55-60, 85-90
    
    LOSS_TYPES = [LOSS_TYPE] * len(MODELS_LIST)
    OPTIMIZERS_TYPES = [OPTIMIZER_TYPE] * len(MODELS_LIST)

    EXTRA = "MM" # ""
    model_files = [
        complete_model_name(
            m.__name__,
            optimizer_type=OPTIMIZER_TYPE,
            loss_type=LOSS_TYPE,
            dataset_type=DATA_TYPE,
            image_type=IMAGE_TYPE,
            isDeNoising=DENOISING,
            extra = EXTRA,
        ) for m, LOSS_TYPE, OPTIMIZERS_TYPE in zip(MODELS_LIST, LOSS_TYPES, OPTIMIZERS_TYPES)
    ]

    MODEL_PATHS = [os.path.join(MODEL_PATH, mf) for mf in model_files]
    OPTIMIZERS = [select_optimizer[opt](m) for m,opt in zip(MODELS_LIST, OPTIMIZERS_TYPES)]
    LOSS_FUNCTIONS = [select_loss[l] for l in LOSS_TYPES] # change

    INFO("MODEL, OPTIM, LOSS ARE READY")
    
    
    # Automated
    trainer = AutoEncoder_Trainer(
                     models_list = MODELS_LIST,
                     model_paths = MODEL_PATHS,
                     train_loader = train_loader,
                     val_loader = val_loader,
                     optimizer = OPTIMIZERS,
                     loss_criterion = LOSS_FUNCTIONS,
                     epochs = EPOCHS,
                     status_rate = 10,
                     lr_scheduler_params = {"factor": 0.75, "patience": 4, "threshold": 5e-5},
                     useHalfPrecision = False,
                     run_status_file = "run_status_clstm.txt",
                     destructAll = True,
                     useGPU = True,
                     debug = True
                )
    INFO("STARTING THE TRAINING")
    trainer.train()

    # 2. Testing
    test_data = MovingMNIST(parent_path = DATA_PATH, isTrain = False)
    INFO("TESTING DATA READY")

    for ae_model in trainer.autoencoder_models:
        print("-"*40)
        print(ae_model.model_file)
        tester = AutoEncoder_Tester(
            model = ae_model.model,
            dataset = test_data,
            model_file = ae_model.model_file,
            stackFrames = stackFrames,
            save_vis = False,
            useGPU = True
        )
        results = tester.test()
        print("-"*40)
        del tester
    
    trainer.clear_memory()
    del trainer
    INFO("** Sucessfully Completed Execution **")
