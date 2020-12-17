import sys
sys.path.append("..")

from test_autoencoder import *

def test_model(
    dataset_type,
    model,
    model_path,
    tester_kwargs = {
        "stackFrames": 1,
        "save_vis": True,
        "n_seed": 8,
        "useGPU": True
    }
    
):
    test_data, channels = select_dataset(dataset_type, isTrain = False)
    load_model(model, model_path)

    print("-"*40)
    model.isTrain = False
    tester = AutoEncoder_Tester(
        model = model,
        dataset = test_data,
        model_file = model_path,
        **tester_kwargs
    )
    results = tester.test()
    print("="*40)