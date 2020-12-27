import argparse
import sys
sys.path.append("..")

# from general.utils import INFO

# Import AutoEncoder trainer and tester scripts
from train_autoencoder import *
from test_autoencoder import *

# Import all available models
from C2D_Models import *
from PatchWise.models_PatchWise import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Parameters to run the training")
    parser.add_argument("--data_path", type = str, help="Path to read data from")
    parser.add_argument("--model_path", type = str, help="Path to store the models")
    args = parser.parse_args()
    
    # Editable
    IMAGE_SIZE = 224
    EPOCHS = 300
    BATCH_SIZE = 128
    IMAGE_TYPE = "normal"
    MODEL_PATH = args.model_path
    if not os.path.exists(MODEL_PATH): os.mkdir(MODEL_PATH)
    DATA_PATH = os.path.join(args.data_path, "VAD_Datasets")
    OPTIMIZER_TYPE = "adam"
    LOSS_TYPE = "mse"
    DENOISING = False
    PATCH_WISE = False
    STACKED = False
    
    isVideo = False
    stackFrames = 1
    asImages = True
    if isVideo or STACKED:
        stackFrames = 16
        asImages = False
        
    # Manual
    DATA_TYPE = "ucsd2" 
    
    # 1. Training
    train_data, CHANNELS = select_dataset(
        dataset = DATA_TYPE,
        parent_path = DATA_PATH,
        isTrain = True,
        asImages = asImages,
        image_size = IMAGE_SIZE,
        image_type = IMAGE_TYPE,
        n_frames = 16,
        frame_strides = [1,2,4,8,16],
        sample_stride = 1,
    )

    train_loader, val_loader = get_data_loaders(train_data, batch_size = BATCH_SIZE)
    INFO("TRAINING DATA READY")
    
    MODELS_LIST = [
        C2D_AE_224(channels = CHANNELS),
        ConvAttentionWrapper(C2D_AE_224(channels = CHANNELS)),

        C2D_AE_224(channels = CHANNELS),
        ConvAttentionWrapper(C2D_AE_224(channels = CHANNELS)),

        C2D_AE_224(channels = CHANNELS, add_dropouts = True),
        ConvAttentionWrapper(C2D_AE_224(channels = CHANNELS, add_dropouts = True)),

        C2D_AE_224(channels = CHANNELS, add_sqzex = True),
        ConvAttentionWrapper(C2D_AE_224(channels = CHANNELS, add_sqzex = True)),
    ]
    
    # weighted
    # psnr
    # dropout
    # se

    model_files = [
        "%s_%s_%s_%s_%s_E%03d_BS%03d"%(m.__name__, DATA_TYPE.upper(), IMAGE_TYPE, OPTIMIZER_TYPE, LOSS_TYPE, EPOCHS, BATCH_SIZE) for m in MODELS_LIST
    ]
    
    if DENOISING: model_files = [mf + "_DeNoising" for mf in model_files]
    if PATCH_WISE: model_files = [mf + "_PatchWise" for mf in model_files]
    if STACKED: model_files = [mf + "_Stacked" for mf in model_files]
    
    MODEL_PATHS = [os.path.join(MODEL_PATH, mf) for mf in model_files]
    OPTIMIZERS = [select_optimizer[OPTIMIZER_TYPE](m) for m in MODELS_LIST]
    # LOSS_FUNCTIONS = [select_loss[LOSS_TYPE] for m in MODELS_LIST]

    model_files[0] += "_WEIGHTED"
    model_files[1] += "_WEIGHTED"
    model_files[2] += "_PSNR"
    model_files[3] += "_PSNR"
    
    LOSS_FUNCTIONS = [select_loss["weighted"], select_loss["weighted"], select_loss["psnr"], select_loss["psnr"], select_loss["mse"], select_loss["mse"], select_loss["mse"], select_loss["mse"]] # change

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
                     lr_scheduler_params = {"factor": 0.75, "patience": 4, "threshold": 1e-6},
                     useHalfPrecision = False,
                     run_status_file = "run_status_c2d.txt",
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
        n_frames = 16,
        frame_strides = [1,2,4,8,16],
        sample_stride = 1,
    )
    INFO("TESTING DATA READY")

    for ae_model in trainer.autoencoder_models:
        print("-"*40)
        print(ae_model.model_file)
        if "vae" in ae_model.model_file.lower():
            try:
                ae_model.isTrain = False
                if "attention" in ae_model.model_file.lower():
                    ae_model.model.isTrain = False
            except:
                pass
        tester = AutoEncoder_Tester(
            model = ae_model.model,
            dataset = test_data,
            model_file = ae_model.model_file,
            stackFrames = stackFrames,
            save_vis = True,
            useGPU = True
        )
        results = tester.test()
        print("-"*40)
        del tester
    
    trainer.clear_memory()
    del trainer
    INFO("** Sucessfully Completed Execution **")
