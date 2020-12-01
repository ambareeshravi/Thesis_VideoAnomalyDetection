import argparse
import sys
sys.path.append("..")

from general import *

# Import AutoEncoder trainer and tester scripts
from train_autoencoder_PL import *
from test_autoencoder import *

# Import all available models
from C2D_Models import *
from PatchWise.models_PatchWise import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Parameters to run the training")
    parser.add_argument("--data_path", type = str, help="Path to read data from")
    parser.add_argument("--model_path", type = str, help="Path to store the models")
    parser.add_argument("--nodes", type = int, default = 1, help = "Number of GPU nodes")
    args = parser.parse_args()
    
    # Editable
    IMAGE_SIZE = 128
    EPOCHS = 300
    BATCH_SIZE = 128
    IMAGE_TYPE = "normal"
    MODEL_PATH = args.model_path
    create_directory(MODEL_PATH)
    DATA_PATH = os.path.join(args.data_path, "VAD_Datasets")
    OPTIMIZER_TYPE = "adam"
    LOSS_TYPE = "mse"
    DENOISING = False
    STACKED = False
    PATCH_WISE = False
    
    isVideo = False
    stackFrames = 1
    asImages = True
    if isVideo:
        stackFrames = 16
        asImages = False
        
    # Manual
    DATA_TYPE = "ucsd2" 
    
    # PL Params
    PRECISION = 32 #16
    GPUS = -1
    GRAD_CLIP_VAL = 0
    NODES = args.nodes
    
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
    
    # Manual
    model = C2D_AE_128_3x3(channels = CHANNELS)
    
    model_file = "PL_%s_%s_%s_%s_E%03d_BS%03d"%(model.__name__, IMAGE_TYPE, OPTIMIZER_TYPE, LOSS_TYPE, EPOCHS, BATCH_SIZE)
    if DENOISING: model_file += "_DeNoising"
    if STACKED: model_file += "_Stacked"
    if PATCH_WISE: model_file += "_PatchWise"
        
    MODEL_SAVE_PATH = os.path.join(MODEL_PATH, model_file)
    create_directory(MODEL_SAVE_PATH)
    lm_model = AutoEncoderLM(
        model,
        MODEL_SAVE_PATH,
        LOSS_TYPE,
        OPTIMIZER_TYPE,
        default_learning_rate = 1e-3,
        max_epochs = 300,
        status_rate = 25,
        lr_scheduler_kwargs = {
            'factor': 0.75,
            'patience': 5,
            'threshold': 1e-4,
            'verbose': True
         }
    )
    INFO("MODEL, OPTIM, LOSS ARE READY")
    
    # Automated
    callbacks_list = [
        EpochChange(),
#         EarlyStopping('validation_loss', min_delta = 1e-8, patience=15, verbose=True)
    ]
    
    trainer = Trainer(
        default_root_dir = MODEL_SAVE_PATH,
        gpus=GPUS,
        num_nodes = NODES,
        accelerator='ddp',
        min_epochs = 50,
        max_epochs = EPOCHS,
        gradient_clip_val = GRAD_CLIP_VAL,
        precision = PRECISION,
        callbacks = callbacks_list,
        progress_bar_refresh_rate = 0,
        weights_summary = None,
        auto_lr_find=True
    )
    trainer.tune(model = lm_model, train_dataloader = train_loader)
    INFO("STARTING THE TRAINING")
    trainer.fit(model = lm_model, train_dataloader=train_loader, val_dataloaders=val_loader)

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
    
    # Changed for PL module
    ae_model = trainer.model
    print("-"*40)
    print(ae_model.model_file)
    tester = AutoEncoder_Tester(
        model = ae_model.model,
        dataset = test_data,
        patchwise = PATCH_WISE,
        stacked = STACKED,
        useGPU = True
    )
    results = tester.test(
        stackFrames = stackFrames,
        isVideo = isVideo,
        save_as = ".pkl".join(ae_model.model_file.split(".pth.tar"))
    )
    print("-"*40)
    try:
        del trainer, tester
        gc.collect()
    except:
        pass
    INFO("** Sucessfully Completed Execution **")
