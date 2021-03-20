from .all_imports import *

from sklearn.metrics import make_scorer, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from torchvision.utils import make_grid, save_image
import subprocess

def execute_bash(command):
    return subprocess.getoutput(command)

def load_json(file):
    if ".json" not in file: file += ".json"
    with open(file, "r") as f:
        contents = json.load(f)
    return contents

def dump_json(contents, file):
    if ".json" not in file: file += ".json"
    with open(file, "w") as f:
        json.dump(contents, f)
    return True

def load_pickle(file):
    if ".pkl" not in file: file += ".pkl"
    with open(file, "rb") as f:
        contents = pkl.load(f)
    return contents
    
def dump_pickle(contents, file):
    if ".pkl" not in file: file += ".pkl"
    with open(file, "wb") as f:
        pkl.dump(contents, f)
    return True

def join_paths(paths):
    path = ""
    for tag in paths:
        path = os.path.join(path, tag)
    return path

def read_directory_contents(directory):
    if "*" not in directory: directory = join_paths([directory, "*"])
    return sorted(glob(directory))

def read_image(image_path, asGray = False):
    if isinstance(image_path, np.ndarray): image = Image.fromarray(image_path)
    else: image = Image.open(image_path)
    if asGray: return image.convert('L')
    return image

def image_cwh(image):
    return image.transpose(0,1).transpose(1,2)

def image_whc(image):
    return image.transpose(1,2).transpose(0,1)

def image_1(image):
    return image/255.

def image_255(image):
    if image.max() > 1: return image.astype(np.uint8)
    return (image*255).astype(np.uint8)

def image_int(image):
    if not image.max() >1: return image_255(image)
    return np.uint8(image)

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

def extend_gray(image_array):
    if len(image_array.shape) < 3: image_array = np.expand_dims(image_array, axis = -1)
    return image_array
    
def array_3channels(image_array):
    image_array = extend_gray(image_array)
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
        plt.plot(epochs, history["validation_loss"], label = "Validation Loss")
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

def flatten_2darray(array):
    return np.array([item for sublist in array for item in sublist])

def moving_average(x, window = 3):
    averaged = np.array([np.mean(x[idx:(idx+window)]) for idx in range(len(x[(window-1):]))])
    residue = x[:(len(x) - len(averaged))]
    return np.concatenate((residue, averaged))
    
def add_noise(img, var = 0.1):
    return img + (torch.randn(img.size(), device = img.device) * var * torch.randint(0, 2, img.shape, device = img.device))

def _init_fn(worker_id):
    np.random.seed(int(MANUAL_SEED))
                   
def get_data_loaders(
    data,
    batch_size = 64,
    val_split = 0.1,
    num_workers = 4,
):

    split_point = int((1-val_split) * len(data))
    train_data, val_data = torch.utils.data.random_split(data, [split_point, len(data)-split_point])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle=True,  num_workers = num_workers, pin_memory=True, worker_init_fn=_init_fn)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size = batch_size, shuffle=False, num_workers = num_workers, pin_memory=True, worker_init_fn=_init_fn)
    return train_loader, val_loader

def eta(epoch, epochs, epoch_time):
    eta_hours = (epoch_time * (epochs - epoch)) / 3600
    eta_ts = datetime.now() + timedelta(hours = eta_hours)
    eta_string = "Maximum estimated Time: %0.2f hours | Will be completed by: %s"%(eta_hours, str(eta_ts)[:16]) + "\n" + "-"*60 + "\n"
    print(eta_string)
    return eta_string
    
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

def create_directory(path):
    if not os.path.exists(path): os.mkdir(path)
        
def getModelFileName(save_path):
    return join_paths([save_path, os.path.split(save_path)[-1] + ".pth.tar"])

def moving_average2(values, window):
    weights = np.repeat(1.0, window)/window
    smas = np.convolve(values, weights, 'valid')
    return smas

def calculate_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

def noise_input(images, NOISE_RATIO = 0.1):
    return images * (1 - NOISE_RATIO) + torch.rand(images.size()) * NOISE_RATIO

def thresholdJ(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]
    return best_thresh

def scores2labels(y_scores, threshold):
    labels = y_scores.copy()
    labels[labels>=threshold] = 1
    labels[labels<threshold] = 0
    return labels

def get_video_frame_count(source):
    cap_flag = False
    if isinstance(resize_to, str):
        cap_flag = True
        source = cv2.VideoCapture(source)
    n_frames = source.get(7)
    if cap_flag:
        source.release()
        del source
    return n_frames

def video2frames(
    video_path:str,
    resize_to = False,
    read_rate:int = 1,
    read_fps = None,
    save_path = False,
    return_count = False,
    start_at:int = 0,
    stop_at:int = 0,
    init_read_count = 0,
    extension:str = ".png"
):
    cap = cv2.VideoCapture(video_path)
    frames = list()
    frame_count = 0
    read_count = init_read_count
    
    assert read_rate == 1 or read_fps == None, "[ERROR]: Use either read rate or read fps. Not both"
    
    w = int(cap.get(3))
    h = int(cap.get(4))
    fps = int(cap.get(5))
    n_frames = int(cap.get(7))
    
    check_rate = read_rate
    
    if read_fps != None:
        try: assert fps % read_fps == 0, "[ERROR]: The resultant number of frames will be imperfect"
        except:
            while fps%read_fps != 0:
                read_fps += 1
            print("Changed read_fps to", read_fps)
        check_rate = fps / read_fps 
    
    while cap.isOpened():
        isRead, frame = cap.read()
        if not isRead: break
        frame_count += 1
        if (cap.get(0)//1000) < start_at: continue
        if frame_count % check_rate != 0: continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if resize_to: cv2.resize(frame, resize_to)
        if save_path:
            cv2.imwrite("%s/%06d%s"%(save_path, read_count, extension), frame)
        if not return_count: frames.append(frame)
        read_count += 1
        if stop_at > 0 and (cap.get(0)//1000 > stop_at): break
        
    cap.release()
    del cap
    if return_count or save_path: return read_count
    else: return frames
    
def most_common(lst):
    return max(set(lst), key=lst.count)

def complete_model_name(
    model_type:str,
    optimizer_type:str,
    loss_type:str,
    dataset_type:str,
    image_type:str,
    isDeNoising:bool,
    extra:str="",
):
    model_type += "_DeNoising" if isDeNoising else ""
    model_name = "%s_%s_%s_%s_%s"%(model_type.upper(), optimizer_type.upper(), loss_type.upper(), dataset_type.upper(), image_type.upper())
    if len(extra) > 0: model_name += "-{%s}"%(extra)
    return model_name

class CustomLogger:
    def __init__(
        self,
        file_path:str,
        extension:str = ".txt",
    ):
        self.file_path = file_path
        if extension not in self.file_path: self.file_path += extension
        self.f = open(self.file_path, "w")
        
    def print(self, *args, sep = " "):
        self.f.write(sep.join([str(a) for a in args]) + "\n")
        self.f.flush()
    
    def __del__(self):
        self.f.close()
        
def scale(x, t_min = 1e-1, t_max = 1):
    return (((x - x.min())/(x.max()-x.min())) * (t_max - t_min)) + t_min