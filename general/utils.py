from .all_imports import *

from torchvision.utils import make_grid, save_image

def join_paths(paths):
    path = ""
    for tag in paths:
        path = os.path.join(path, tag)
    return path

def read_directory_contents(directory):
    return sorted(glob(directory))

def read_image(image_path, asGray = False):
    if asGray: return Image.open(image_path).convert('L')
    return Image.open(image_path)

def image_1(image):
    return image/255.

def image_255(image):
    return (image*255).astype(np.uint8)

def array_to_image(image_array):
    return Image.fromarray(image_array)

def array_to_gray(image_array, returnArray = True):
    if image_array.shape[-1] == 3:
        x = array_to_image(image_array).convert('L')
        if returnArray: return np.array(x)
        return x
    else:
        return image_array

def shrink_gray(image_array):
    assert len(image_array.shape) == 3 and (image_array.shape[-1] == 1), "image not in required format: h x w x 1"
    return image_array.squeeze(axis = -1)

def array_3channels(image_array):
    if len(image_array.shape) < 3: image_array = np.expand_dims(image_array, axis = -1)
    if image_array.shape[-1] == 1: image_array = np.repeat(image_array, 3, axis = -1)
    return image_array

def save_model(model, model_name):
    if ".tar" not in model_name: model_name += ".tar"
    torch.save({'state_dict': model.state_dict()}, model_name)

def load_model(model, model_name):
    if ".tar" not in model_name: model_name += ".tar"
    checkpoint = torch.load(model_name, map_location = 'cpu')
    model.load_state_dict(checkpoint['state_dict'])

def plot_stat(history, title, save_path):
    try:
        epochs = list(range(len(history["train_loss"])))
        plt.plot(epochs, history["train_loss"], label = "Train Loss")
        plt.plot(epochs, history["val_loss"], label = "Validation Loss")
        plt.legend()
        plt.xlabel("Number of epochs")
        plt.ylabel("Loss")
        plt.title(title)
        plt.savefig(os.path.join(save_path, "training_stats.png"), dpi = 100, bbox_inches='tight')
        plt.clf()
        try: plt.close()
        except: pass
    except Exception as e:
        print(e)
        
def to_cpu(x):
    return x.detach().cpu()
    
def tensor_to_numpy(x):
    return to_cpu(x).numpy()

def normalize_error(x):
    return (x - x.min()) / (x.max() - x.min())

def flatten_2darray(array):
    return np.array([item for sublist in array for item in sublist])

def moving_average(x, window = 3):
    averaged = np.array([np.mean(x[idx:(idx+window)]) for idx in range(len(x[(window-1):]))])
    residue = x[:(len(x) - len(averaged))]
    return np.concatenate((residue, averaged))
    
def add_noise(img, var = 0.1, device = torch.device("cpu")):
    noise = torch.randn(img.size(), device = device) * var
    noisy_img = img + noise
    return noisy_img

def get_data_loaders(
    data,
    batch_size = 64,
    val_split = 0.1,
    num_workers = 4,
):

    split_point = int((1-val_split) * len(data))
    train_data, val_data = torch.utils.data.random_split(data, [split_point, len(data)-split_point])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle=True,  num_workers = num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle=False, num_workers = num_workers, pin_memory=True)
    return train_loader, val_loader

def eta(epoch, epochs, epoch_time):
    eta_hours = (epoch_time * (epochs - epoch)) / 3600
    eta_ts = datetime.now() + timedelta(hours = eta_hours)
    print("Estimated Time %0.2f hours | Will be completed by %s:"%(eta_hours, str(eta_ts)[:16]))
    
def get_patches(images, patch_size = 64, overlap = 32):
    patches = list()
    coord_pairs = [[col_idx, row_idx] for col_idx in range(0, images.shape[-1], overlap) for row_idx in range(0, images.shape[-2], overlap)]
    for x,y in coord_pairs:
        image_crop = images[..., x:(x+patch_size), y:(y+patch_size)]
        if image_crop.shape[-2] != patch_size or image_crop.shape[-1] != patch_size: continue
        patches.append(image_crop)
    return torch.cat(patches, dim = 0)

def merge_patches(image_tensor, n_patches = 4):
    # assumes that get_patches use patch_size and overlap as 64
    if len(image_tensor.shape) < 5:
        return torch.stack([make_grid(image_tensor[idx:(idx+n_patches)], nrow = 2, padding = 0) for idx in range(0, len(image_tensor), n_patches)])
    else:
        return torch.stack([torch.stack([make_grid(image_tensor[:,:,ts,:,:][idx:(idx+n_patches)], nrow = 2, padding = 0) for idx in range(0, len(image_tensor[:,:,ts,:,:]), n_patches)]) for ts in range(image_tensor.shape[-3])]).permute(1,2,0,3,4)

def INFO(string):
    print("-"*40)
    print("[INFO]:", string)
    print("-"*40)

def ERROR(error, error_code = ""):
    print("-"*40)
    print("[ERROR]", error_code, ":", error)
    print("-"*40)
    
class TimeIt:
    def __init__(self, unit = "seconds"):
        self.unit = unit
    
    def start(self):
        self.starting_time = time()
    
    def end(self):
        self.ending_time = time()
        
    def time_elapsed(self):
        if "seconds" in self.unit:
            return (self.ending_time - self.starting_time)
        elif "minutes" in self.unit:
            return (self.ending_time - self.starting_time)/60
        elif "hours" in self.unit:
            return (self.ending_time - self.starting_time)/3600
        
def read_txt(file_path):
    with open(file_path, "r") as f:
        contents = f.read()
    return contents