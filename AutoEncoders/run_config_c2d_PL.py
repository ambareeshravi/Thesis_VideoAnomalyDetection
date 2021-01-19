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
    EPOCHS = 200
    BATCH_SIZE = 128
    IMAGE_TYPE = "normal"
    MODEL_PATH = args.model_path
    create_directory(MODEL_PATH)
    DATA_PATH = os.path.join(args.data_path, "VAD_Datasets")
    OPTIMIZER_TYPE = "adam"
    LOSS_TYPE = "mse"
    DENOISING = False
    PATCH_WISE = False
    STACKED = False
    
    isVideo = False
    stackFrames = 1
    asImages = True
    if isVideo:
        stackFrames = 16
        asImages = False
        
    # Manual
    DATA_TYPE = "ucsd1" 
    
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
    model = C2D_AE_128_3x3_VAE(channels = CHANNELS)
    model_file = complete_model_name(
        model.__name__,
        optimizer_type=OPTIMIZER_TYPE,
        loss_type=LOSS_TYPE,
        dataset_type=DATA_TYPE,
        image_type=IMAGE_TYPE,
        isDeNoising=DENOISING
    )
    
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
        max_epochs = EPOCHS,
        status_rate = 25,
        lr_scheduler_kwargs = {
            'factor': 0.75,
            'patience': 4,
            'threshold': 5e-5,
            'verbose': True
         }
    )
    INFO("MODEL, OPTIM, LOSS ARE READY")
    
    # Automated
    callbacks_list = [
        EpochChange(),
        EarlyStopping('validation_loss', min_delta=1e-6, patience=16, mode="min", verbose=True),
        GPUStatsMonitor()
    ]
    
    trainer = Trainer(
        default_root_dir = MODEL_PATH,
        gpus=GPUS,
        num_nodes = NODES,
        accelerator='ddp',
        min_epochs = 75,
        max_epochs = EPOCHS,
#         gradient_clip_val = GRAD_CLIP_VAL,
        precision = PRECISION,
        callbacks = callbacks_list,
        accumulate_grad_batches={50: 2, 100: 4},
        progress_bar_refresh_rate = 0,
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
    
    print("-"*40)
    model_file = getModelFileName(MODEL_SAVE_PATH)
    load_model(model, model_file)
    print(model_file)
    if "vae" in model_file.lower():
        try:
            model.isTrain = False
            if "attention" in model_file.lower():
                model.model.isTrain = False
        except:
            pass
    tester = AutoEncoder_Tester(
        model = model,
        dataset = test_data,
        model_file = model_file,
        stackFrames = stackFrames,
        save_vis = True,
        useGPU = True
    )
    results = tester.test()
    print("-"*40)
    try:
        del trainer, tester
        gc.collect()
    except:
        pass
    INFO("** Sucessfully Completed Execution **")