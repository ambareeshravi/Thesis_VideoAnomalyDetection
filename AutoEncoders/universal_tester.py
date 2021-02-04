import sys
sys.path.append("..")

from test_autoencoder import *
from pprint import pprint

def test_model(
    dataset_type,
    model,
    model_path,
    tester_kwargs = {
        "stackFrames": 1,
        "save_vis": True,
        "n_seed": 8,
        "applyFilter": False,
        "useGPU": True
    },
    dataset_kwargs ={
        'image_size': 128,
        'image_type': 'normal'
    },
    return_results = False
):
    test_data, channels = select_dataset(dataset_type, isTrain = False, **dataset_kwargs)
    load_model(model, model_path)

    print("-"*40)
    model.isTrain = False
    tester = AutoEncoder_Tester(
        model = model,
        dataset = test_data,
        model_file = model_path,
        **tester_kwargs
    )
    results = tester.test(return_results)
    print("="*40)
    try: del test_data, tester
    except: pass
    if return_results: return results
    else: return True