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
    IMAGE_SIZE = 128
    EPOCHS = 300
    BATCH_SIZE = 32
    IMAGE_TYPE = "normal"
    MODEL_PATH = args.model_path
    if not os.path.exists(MODEL_PATH): os.mkdir(MODEL_PATH)
    DATA_PATH = os.path.join(args.data_path, "VAD_Datasets")
    OPTIMIZER_TYPE = "adam"
    LOSS_TYPE = "mse"
    DENOISING = False
    
    isVideo = True
    stackFrames = 1
    asImages = True
    if isVideo:
        stackFrames = 8
        asImages = False
        
    # Manual
    DATA_TYPE = "ucsd1" 
    
    # 1. Training
    train_data, CHANNELS = select_dataset(
        dataset = DATA_TYPE,
        parent_path = DATA_PATH,
        isTrain = True,
        asImages = asImages,
        image_size = IMAGE_SIZE,
        image_type = IMAGE_TYPE,
        n_frames = stackFrames,
        frame_strides = [1,2,4,8,16],
        sample_stride = 1,
    )

    train_loader, val_loader = get_data_loaders(train_data, batch_size = BATCH_SIZE)
    INFO("TRAINING DATA READY")
    
    MODELS_LIST = [
        CRNN_AE(image_size = IMAGE_SIZE, channels = CHANNELS),
        CLSTM_AE(image_size = IMAGE_SIZE, channels = CHANNELS)
    ]
    
    LOSS_TYPES = [LOSS_TYPE] * len(MODELS_LIST)
#     LOSS_TYPES = ["weighted", "weighted", "psnr", "psnr", "mse", "mse", "mse", "mse"]
    OPTIMIZERS_TYPES = [OPTIMIZER_TYPE] * len(MODELS_LIST)
#     OPTIMIZERS_TYPES = ["adam", "adagrad", "sgd"]

    model_files = [
        complete_model_name(
            m.__name__,
            optimizer_type=OPTIMIZER_TYPE,
            loss_type=LOSS_TYPE,
            dataset_type=DATA_TYPE,
            image_type=IMAGE_TYPE,
            isDeNoising=DENOISING
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
                     status_rate = 25,
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
    test_data, CHANNELS = select_dataset(
        dataset = DATA_TYPE,
        parent_path = DATA_PATH,
        isTrain = False,
        asImages = True,
        image_size = IMAGE_SIZE,
        image_type = IMAGE_TYPE,
        n_frames = stackFrames,
        frame_strides = [2,4,8,16],
        sample_stride = 1,
    )
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
