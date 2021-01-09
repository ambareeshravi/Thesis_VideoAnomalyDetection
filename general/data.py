from .all_imports import *
from .utils import *
from .cv2_processing import OpticalFlow, BackgroundSubtraction

from skimage import filters

NORMAL_LABEL = 1
ABNORMAL_LABEL = 0

# Handles Image data
class ImagesHandler:
    def __init__(self, data_transforms):
        self.data_transform = data_transforms
        self.select_image_type()
        
    def select_image_type(self,):
        image_type = self.image_type.lower()
        if "flow" in image_type:
            self.image_type = "flow"
            self.n_frames += 1
            returnMag = True
            if "clr" in image_type: returnMag = False
            self.opt_flow = OpticalFlow(returnMag = returnMag)
            self.read_frames = self.read_flow
        elif "difference" in image_type:
            self.image_type = "difference"
            self.n_frames += 1
            self.read_frames = self.read_difference
        elif "mask" in image_type:
            self.image_type = "mask"
            self.read_frames = self.read_mask
        elif "background" in image_type:
            self.image_type = "background"
            self.read_frames = self.read_background
        elif "grayscale" in image_type:
            self.image_type = "grayscale"
            self.read_frames = self.read_gray
        else:
            self.image_type = "normal"
            self.read_frames = self.read_normal
            
    def stride_through_frames(self, files):
        files = np.asarray(files)
        indices = np.array(range(0, len(files), self.sample_stride))
        return files[indices], indices
        
    def read_normal(self, files):
        return [self.data_transform(read_image(image_path)) for image_path in files]
    
    def read_gray(self, files):
        return [self.data_transform(read_image(image_path, asGray = True)) for image_path in files]
        
    def read_flow(self, files):
        frames = [read_image(image_path) for image_path in files]
        frame_arrays = [np.array(frame) for frame in frames]
        frame_flows = [self.opt_flow.get_optical_flow(frame_array, frame_arrays[idx+1]) for idx, frame_array in enumerate(frame_arrays[:-1])]
        flow_combined = [torch.cat((self.data_transform(Image.fromarray(image_255(frame))), self.data_transform(Image.fromarray(image_255(flow)))), dim = 0) for idx, (frame, flow) in enumerate(zip(frame_arrays[:-1], frame_flows))]
        return flow_combined
        
    # difference is useless
    def get_difference(self, frame1, frame2):
        difference = image_255(np.asarray(frame2.convert('L')) - np.asarray(frame1.convert('L')))
        return Image.fromarray(np.concatenate((extend_gray(frame1), extend_gray(difference)), axis = -1))

    def read_difference(self, files):
        frames = [read_image(image_path) for image_path in files]
        return [self.data_transform(self.get_difference(frames[idx-1], frames[idx])) for idx in range(1, len(frames))]
    
    # thresholds only white colors -> useless too
    def add_mask(self, original_frame):
        gray = np.asarray(original_frame.convert('L'))
        original_frame = extend_gray(np.asarray(original_frame, dtype = np.float32)/255.)
        threshold = filters.threshold_otsu(gray, nbins=256)
        mask = gray < threshold
        mask = extend_gray(np.asarray(mask * 1.0))
        mask = image_255(mask)
        combined = image_255(np.array(original_frame) * mask)
        return Image.fromarray(combined.squeeze())
    
    def read_mask(self, files):
        return [self.data_transform(self.add_mask(read_image(image_path))) for image_path in files]
    
    def read_background(self, files):
        bs = BackgroundSubtraction()
        frames = [read_image(image_path) for image_path in files]
        frames_arrays = [np.array(frame) for frame in frames]
        combined = [np.concatenate((extend_gray(frame_array), extend_gray(bs(frame_array))), axis = -1) for frame_array in frames_arrays]
        del bs
        return [self.data_transform(Image.fromarray(cmbd_img)) for cmbd_img in combined]        

# Handles Video data
class VideosHandler:
    def __init__(self, data_transforms):
        self.data_transform = data_transforms
        
    def read_video_frames(self, files):
        return torch.stack(self.read_frames(files)).transpose(0,1)
    
    def select_indices(self, num_files):
        segment_indices = np.empty((self.n_frames), dtype = np.int16)
        for idx in range(0, num_files, self.sample_stride):
            for frame_stride in self.frame_strides:
                candidate_indices = np.array(range(idx, num_files, frame_stride)).astype(np.int16)
                frames_set = candidate_indices[:((len(candidate_indices)//self.n_frames)*self.n_frames)].reshape(-1, self.n_frames)
                segment_indices = np.vstack((segment_indices, frames_set))
        segment_indices = np.unique(segment_indices[1:], axis = 0)
        return segment_indices
    
    def read_videos(self, files):
        files = np.array(files)
        num_files = len(files)
        segment_indices = self.select_indices(num_files)
        video_frames = list()
        for indices in segment_indices:
            video_frames.append(self.read_video_frames(files[indices]))
        return video_frames, segment_indices

# Common Attributes
class Attributes:
    def __init__(self):
        pass
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# UCSD Data
class UCSD(ImagesHandler, VideosHandler, Attributes):
    def __init__(self,
                 parent_path = "../../../datasets/VAD_Datasets/",
                 dataset_type = 1,
                 isTrain = True,
                 asImages = True,
                 image_size = 128,
                 image_type = "normal",
                 n_frames = 16,
                 frame_strides = [1,2,4,8,16],
                 sample_stride = 1,
                 useCorrectedAnnotations = False
                ):
        self.__name__ = "UCSD" + str(dataset_type) + "_V2"
        self.isTrain = isTrain
        self.asImages = asImages
        self.image_size = image_size
        self.image_type = image_type
        self.n_frames = n_frames
        self.frame_strides = frame_strides
        self.sample_stride = sample_stride
        
        if isinstance(self.image_size, int): self.image_size = (self.image_size, self.image_size)
        
        transforms_list = list()
        if dataset_type == 1:
            transforms_list += [lambda x: transforms.F.crop(x, 14, 8, 144, 204)]
        else:
            transforms_list += [lambda x: transforms.F.crop(x, 80, 0, 160, 360)]
        
        if self.isTrain:
            transforms_list += [
#                 transforms.ColorJitter(brightness=0.9, contrast=0.9),
#                 transforms.RandomAffine(3, translate=None, scale=None, shear=None, resample=0, fillcolor=0),
#                 transforms.RandomHorizontalFlip(p=0.25),
            ]
        transforms_list += [
            transforms.Resize((self.image_size[0], self.image_size[1])),
            transforms.ToTensor()
        ]
        data_transforms = transforms.Compose(transforms_list)
        
        ImagesHandler.__init__(self, data_transforms)
        VideosHandler.__init__(self, data_transforms)
        self.data_path = os.path.join(parent_path, "UCSD/UCSDped%s/"%(str(dataset_type)))
        
        if self.isTrain:
            self.parent_directory = join_paths([self.data_path, "Train"])
        else:
            self.asImages = True
            self.sample_stride = 1
            self.parent_directory = join_paths([self.data_path, "Test"])
            if useCorrectedAnnotations:
                self.annotations_file = join_paths([self.parent_directory, "UCSDped"+str(dataset_type)+"_corrected.m"])
            else:
                self.annotations_file = join_paths([self.parent_directory, "UCSDped"+str(dataset_type)+".m"])
            self.temporal_labels = self.get_temporal_labels(self.read_temporal_annotations())
            self.frame_strides = [1]
            
            
        if self.asImages:
            self.read = self.read_image_data
        else:
            self.read = self.read_video_data
        self.create_dataset()
        Attributes.__init__(self)
        
    def read_temporal_annotations(self,):
        with open(self.annotations_file, "r") as f:
            annotation_contents = f.readlines()
        annotation_contents = [line.split("[")[-1].split("]")[0] for line in annotation_contents[1:]]
        temporal_annotations = [[list(map(int, item.split(":"))) for item in ac.split(", ")] for ac in annotation_contents]
        return temporal_annotations
        
    def get_temporal_labels(self, temporal_annotations):
        temporal_labels = list()
        video_folders = read_directory_contents(join_paths([self.parent_directory, "*"]))
        video_folders = [d for d in video_folders if "_gt" not in d and ".m" not in d]
        for ta, folder in zip(temporal_annotations, video_folders):
            x = [NORMAL_LABEL]*len(read_directory_contents(join_paths([folder, "*"])))
            for ann in ta:
                for n in range(ann[0], ann[1] + 1):
                    x[n-1] = ABNORMAL_LABEL
            temporal_labels.append(np.array(x))
        temporal_labels = np.array(temporal_labels)
        return temporal_labels
    
    def read_image_data(self, files_in_dir, idx):
        files_in_dir, indices = self.stride_through_frames(files_in_dir)
        data = self.read_frames(files_in_dir)
        if self.isTrain:
            labels = [NORMAL_LABEL] * len(data)
        else:
            labels = self.temporal_labels[idx][indices].tolist()
        return data, labels
    
    def read_video_data(self, files_in_dir, idx):
        data, segment_indices = self.read_videos(files_in_dir)
        if self.isTrain:
            labels = np.ones(segment_indices.shape, dtype = np.int16).tolist()
        else:
            labels = [self.temporal_labels[idx][si].tolist() for si in segment_indices]
        return data, labels
    
    def create_dataset(self):
        self.data, self.labels = list(), list()
        directories = read_directory_contents(join_paths([self.parent_directory, "*"]))
        directories = [d for d in directories if "_gt" not in d and ".m" not in d]
        for idx, video_directory in tqdm(enumerate(directories)):
            try:
                files_in_dir = read_directory_contents(join_paths([video_directory, "*.tif"]))
                data, labels = self.read(files_in_dir, idx)
                labels = labels[:len(data)]
            except Exception as e:
                print(e)
                continue
            if self.isTrain:
                self.data += data
                self.labels += labels
            else:
                self.data.append(data)
                self.labels.append(labels)
                
# Street Scene data
class StreetScene(ImagesHandler, VideosHandler, Attributes):
    def __init__(self,
                 parent_path = "/media/ambreesh/datasets/",
                 isTrain = True,
                 asImages = True,
                 image_size = 128,
                 image_type = "normal",
                 n_frames = 16,
                 frame_strides = [1,2,4,8,16],
                 sample_stride = 1,
                ):
        self.__name__ = "Street_Scene" + "_V2"
        self.isTrain = isTrain
        self.asImages = asImages
        self.image_size = image_size
        self.image_type = image_type
        self.n_frames = n_frames
        self.frame_strides = frame_strides
        self.sample_stride = sample_stride
        
        if isinstance(self.image_size, int): self.image_size = (self.image_size, self.image_size)
        
        transforms_list = list()
        
        if self.isTrain:
            transforms_list += [
#                 transforms.ColorJitter(brightness=0.9, contrast=0.9),
#                 transforms.RandomAffine(3, translate=None, scale=None, shear=None, resample=0, fillcolor=0),
#                 transforms.RandomHorizontalFlip(p=0.25),
            ]
        transforms_list += [
            transforms.Resize((self.image_size[0], self.image_size[1])),
            transforms.ToTensor()
        ]
        data_transforms = transforms.Compose(transforms_list)
        
        ImagesHandler.__init__(self, data_transforms)
        VideosHandler.__init__(self, data_transforms)
        self.data_path = os.path.join(parent_path, "StreetScene")
        
        if self.isTrain:
            self.parent_directory = join_paths([self.data_path, "Train"])
        else:
            self.asImages = True
            self.sample_stride = 1
            self.parent_directory = join_paths([self.data_path, "Test"])
            self.frame_strides = [1]
            
            
        if self.asImages:
            self.read = self.read_image_data
        else:
            self.read = self.read_video_data
        self.create_dataset()
        Attributes.__init__(self)
        
    def read_image_data(self, files_in_dir, l = NORMAL_LABEL):
        files_in_dir, indices = self.stride_through_frames(files_in_dir)
        data = self.read_frames(files_in_dir)
        labels = [l] * len(data)
        return data, labels
    
    def read_video_data(self, files_in_dir, l = NORMAL_LABEL):
        data, segment_indices = self.read_videos(files_in_dir)
        if l == NORMAL_LABEL:
            labels = np.ones(segment_indices.shape, dtype = np.int16).tolist()
        else:
            labels = np.zeros(segment_indices.shape, dtype = np.int16).tolist()
        return data, labels
    
    def create_dataset(self):
        self.data, self.labels = list(), list()
        directories = read_directory_contents(join_paths([self.parent_directory, "*"]))[:-2]
        if self.isTrain:
            for idx, video_directory in tqdm(enumerate(directories)):
                try:
                    files_in_dir = read_directory_contents(join_paths([video_directory, "*"]))
                    data, labels = self.read(files_in_dir)
                    labels = labels[:len(data)]
                except Exception as e:
                    continue
                self.data += data
                self.labels += labels
        else:
            for idx, test_directory in tqdm(enumerate(directories)):
                image_files = [os.path.split(image_file)[-1] for image_file in read_directory_contents(join_paths([test_directory, "*"])) if ".txt" not in image_file]
                annotation_file = read_directory_contents(join_paths([test_directory, "*.txt"]))[-1]
                df = pd.read_csv(annotation_file, sep = " ", names = ["file_name", "anomaly_id", "x", "y", "w", "h"])[["file_name", "anomaly_id"]]
                normal_files = [join_paths([test_directory, i]) for i in image_files if i not in np.array(df["file_name"])]
                if len(normal_files) < 1:
                    continue
                normal_data, normal_labels = self.read(normal_files, l = NORMAL_LABEL)
                assert len(normal_data) == len(normal_labels), "Mismatch between data and labels"
                for anomaly_id in np.unique(df["anomaly_id"]):
                    try:
                        sub_df = df.where(df["anomaly_id"] == anomaly_id).dropna()
                        anomalous_files = [join_paths([test_directory, i]) for i in sub_df["file_name"]]
                        anomalous_data, anomalous_labels = self.read(anomalous_files, l = ABNORMAL_LABEL)
                        assert len(anomalous_data) == len(anomalous_labels), "Mismatch between data and labels"
                    except:
                        continue
                    self.data.append(normal_data + anomalous_data)
                    self.labels.append(normal_labels + anomalous_labels)
                    
class Avenue(ImagesHandler, VideosHandler, Attributes):
    def __init__(self,
                 parent_path = "../../../datasets/VAD_Datasets/",
                 isTrain = True,
                 asImages = True,
                 image_size = 128,
                 image_type = "normal",
                 n_frames = 16,
                 frame_strides = [1,2,4,8,16],
                 sample_stride = 1,
                ):
        self.__name__ = "Avenue" + "_V2"
        self.isTrain = isTrain
        self.asImages = asImages
        self.image_size = image_size
        self.image_type = image_type
        self.n_frames = n_frames
        self.frame_strides = frame_strides
        self.sample_stride = sample_stride
        
        if isinstance(self.image_size, int): self.image_size = (self.image_size, self.image_size)
        
        transforms_list = list()
        transforms_list += [lambda x: transforms.F.crop(x, 60, 0, 300, 640)]
        
        if self.isTrain:
            transforms_list += [
#                 transforms.ColorJitter(brightness=0.9, contrast=0.9),
#                 transforms.RandomAffine(3, translate=None, scale=None, shear=None, resample=0, fillcolor=0),
#                 transforms.RandomHorizontalFlip(p=0.2),
            ]
        transforms_list += [
            transforms.Resize((self.image_size[0], self.image_size[1])),
            transforms.ToTensor()
        ]
        data_transforms = transforms.Compose(transforms_list)
        
        ImagesHandler.__init__(self, data_transforms)
        VideosHandler.__init__(self, data_transforms)
        self.data_path = os.path.join(parent_path, "Avenue")
        
        if self.isTrain:
            self.parent_directory = join_paths([self.data_path, "Train"])
        else:
            self.asImages = True
            self.sample_stride = 1
            self.parent_directory = join_paths([self.data_path, "Test"])
            self.annotations_file = join_paths([self.data_path, "Avenue_corrected.m"])
            self.temporal_labels = self.get_temporal_labels(self.read_temporal_annotations())
            self.frame_strides = [1]
            
        if self.asImages:
            self.read = self.read_image_data
        else:
            self.read = self.read_video_data
        self.create_dataset()
        Attributes.__init__(self)
        
    def read_temporal_annotations(self,):
        with open(self.annotations_file, "r") as f:
            annotation_contents = f.readlines()
        annotation_contents = [line.split("[")[-1].split("]")[0] for line in annotation_contents[1:]]
        temporal_annotations = [[list(map(int, item.split(":"))) for item in ac.split(", ")] for ac in annotation_contents]
        return temporal_annotations
    
    def get_temporal_labels(self, temporal_annotations):
        temporal_labels = list()
        video_folders = read_directory_contents(join_paths([self.parent_directory, "*"]))
        video_folders = [d for d in video_folders if "_gt" not in d and ".m" not in d]
        for ta, folder in zip(temporal_annotations, video_folders):
            x = [NORMAL_LABEL]*len(read_directory_contents(join_paths([folder, "*"])))
            for ann in ta:
                for n in range(ann[0], ann[1] + 1):
                    x[n-1] = ABNORMAL_LABEL
            temporal_labels.append(np.array(x))
        temporal_labels = np.array(temporal_labels)
        return temporal_labels
    
    def read_image_data(self, files_in_dir, idx):
        files_in_dir, indices = self.stride_through_frames(files_in_dir)
        data = self.read_frames(files_in_dir)
        if self.isTrain:
            labels = [NORMAL_LABEL] * len(data)
        else:
            labels = self.temporal_labels[idx][indices].tolist()
        return data, labels
    
    def read_video_data(self, files_in_dir, idx):
        data, segment_indices = self.read_videos(files_in_dir)
        if self.isTrain:
            labels = np.ones(segment_indices.shape, dtype = np.int16).tolist()
        else:
            labels = [self.temporal_labels[idx][si].tolist() for si in segment_indices]
        return data, labels
    
    def create_dataset(self):
        self.data, self.labels = list(), list()
        directories = read_directory_contents(join_paths([self.parent_directory, "*"]))
        directories = [d for d in directories if "_gt" not in d and ".m" not in d]
        for idx, video_directory in tqdm(enumerate(directories)):
            try:
                files_in_dir = read_directory_contents(join_paths([video_directory, "*.png"]))
                data, labels = self.read(files_in_dir, idx)
                labels = labels[:len(data)]
            except Exception as e:
                print(e)
                continue
            if self.isTrain:
                self.data += data
                self.labels += labels
            else:
                self.data.append(data)
                self.labels.append(labels)
                
class Subway(ImagesHandler, VideosHandler, Attributes):
    def __init__(
        self,
        dataset_type = 0,
        parent_path = "../../../datasets/VAD_Datasets/",
        isTrain = True,
        asImages = True,
        image_size = 128,
        image_type = "normal",
        n_frames = 16,
        frame_strides = [1,2,4,8,16],
        sample_stride = 1,
    ):
        if dataset_type == 0:
            self.dataset_type = 0
            self.dataset_tag = "Entrance"
        else:
            self.dataset_type = 1
            self.dataset_tag = "Exit"
            
        self.__name__ = "%s_%s"%("Subway", self.dataset_tag) + "_V2"
        self.isTrain = isTrain
        self.asImages = asImages
        self.image_size = image_size
        self.image_type = image_type
        self.n_frames = n_frames
        self.frame_strides = frame_strides
        self.sample_stride = sample_stride
        
        if isinstance(self.image_size, int): self.image_size = (self.image_size, self.image_size)
        
        transforms_list = [
            transforms.Grayscale(),
            lambda x: transforms.F.crop(x, 64, 8, 280, 488)
        ]
        
        if self.isTrain:
            transforms_list += [
#                 transforms.ColorJitter(brightness=0.9, contrast=0.9),
#                 transforms.RandomAffine(3, translate=None, scale=None, shear=None, resample=0, fillcolor=0),
#                 transforms.RandomHorizontalFlip(p=0.2),
            ]
        transforms_list += [
            transforms.Resize((self.image_size[0], self.image_size[1])),
            transforms.ToTensor()
        ]
        data_transforms = transforms.Compose(transforms_list)
        ImagesHandler.__init__(self, data_transforms)
        VideosHandler.__init__(self, data_transforms)
        
        self.data_path = os.path.join(parent_path, "%s/%s"%("Subway", self.dataset_tag))
        
        if self.isTrain:
            self.parent_directory = join_paths([self.data_path, "Train"])
        else:
            self.asImages = True
            self.sample_stride = 1
            self.parent_directory = join_paths([self.data_path, "Test/"])
            self.frame_strides = [1]
            self.temporal_labels = self.read_temporal_annotations()
            
        if self.asImages:
            self.read = self.read_image_data
        else:
            self.read = self.read_video_data
            
        self.create_dataset()
        Attributes.__init__(self)
    
    def read_temporal_annotations(self, threshold = 20):
        annotation_file = join_paths([self.data_path, "%s_annotations.json"%(self.dataset_tag.lower())])
        anomalous_frame_numbers = list()
        
        for values in load_json(annotation_file).values():
            anomalous_frame_numbers += values
        
        anomalous_frame_numbers = sorted(np.array(anomalous_frame_numbers))
        frame_ids = np.array([int(os.path.split(i)[-1].split(".")[0]) for i in read_directory_contents(join_paths([self.parent_directory, "*"]))])
        start_frame_number = frame_ids[0]
        adjusted_anomalous_frame_numbers = anomalous_frame_numbers - start_frame_number
        assert np.all(adjusted_anomalous_frame_numbers > 0), "Negative indices found in temporal annotations"
        
        temporal_labels = np.array([NORMAL_LABEL]*len(frame_ids))
        for aafn in adjusted_anomalous_frame_numbers:
            temporal_labels[max(0, aafn-threshold) : min(aafn + threshold, len(temporal_labels))] = ABNORMAL_LABEL
        return np.expand_dims(temporal_labels, axis = 0)
                
    def read_image_data(self, files_in_dir, idx):
        files_in_dir, indices = self.stride_through_frames(files_in_dir)
        data = self.read_frames(files_in_dir)
        if self.isTrain:
            labels = [NORMAL_LABEL] * len(data)
        else:
            labels = self.temporal_labels[idx][indices].tolist()
        return data, labels
    
    def read_video_data(self, files_in_dir, idx):
        data, segment_indices = self.read_videos(files_in_dir)
        if self.isTrain:
            labels = np.ones(segment_indices.shape, dtype = np.int16).tolist()
        else:
            labels = [self.temporal_labels[idx][si].tolist() for si in segment_indices]
        return data, labels
    
    def create_dataset(self):
        self.data, self.labels = list(), list()
        image_files = read_directory_contents(join_paths([self.parent_directory, "*"]))
        data, labels = self.read(image_files, 0)
        assert len(data) == len(labels), "Data and Labels length mismatch"
        if self.isTrain:
            self.data += data
            self.labels += labels
        else:
            if self.dataset_type == 0:
                print("Subway Entrance: Last few frames dropped")
                print(len(data))
                data = data[:-3000]
                labels = labels[:-3000]
                print(len(data))
            self.data.append(data)
            self.labels.append(labels)
            print(len(self.data[0]))
                
class ShangaiTech(ImagesHandler, VideosHandler, Attributes):
    def __init__(
        self,
        parent_path = "../../../datasets/VAD_Datasets/",
        isTrain = True,
        asImages = True,
        image_size = 128,
        image_type = "normal",
        n_frames = 16,
        frame_strides = [1,2,4,8,16],
        sample_stride = 1,
    ):
        self.__name__ = "ShangaiTech" + "_V2"
        self.isTrain = isTrain
        self.asImages = asImages
        self.image_size = image_size
        self.image_type = image_type
        self.n_frames = n_frames
        self.frame_strides = frame_strides
        self.sample_stride = sample_stride
        
        if isinstance(self.image_size, int): self.image_size = (self.image_size, self.image_size)
        
        transforms_list = list()
        
        if self.isTrain:
            transforms_list += [
#                 transforms.ColorJitter(brightness=0.9, contrast=0.9),
#                 transforms.RandomAffine(3, translate=None, scale=None, shear=None, resample=0, fillcolor=0),
#                 transforms.RandomHorizontalFlip(p=0.2),
            ]
        transforms_list += [
            transforms.Resize((self.image_size[0], self.image_size[1])),
            transforms.ToTensor()
        ]
        data_transforms = transforms.Compose(transforms_list)
            
        ImagesHandler.__init__(self, data_transforms)
        VideosHandler.__init__(self, data_transforms)
        
        self.data_path = os.path.join(parent_path, "ShangaiTech")
        
        if self.isTrain:
            self.parent_directory = join_paths([self.data_path, "Train"])
        else:
            self.asImages = True
            self.sample_stride = 1
            self.parent_directory = join_paths([self.data_path, "Test/frames/"])
            self.frame_strides = [1]
            self.temporal_labels = self.read_temporal_annotations()
            
        if self.asImages:
            self.read = self.read_image_data
        else:
            self.read = self.read_video_data
            
        self.create_dataset()
        Attributes.__init__(self)
    
    def read_temporal_annotations(self,):
        annotation_files = read_directory_contents(join_paths([self.data_path, "Test/test_frame_mask/*"]))
        test_annotations = list()
        for af in annotation_files:
            test_annotations.append(np.abs(1-np.load(af))) # the original labels are flipped
        return np.asarray(test_annotations)
        
    def read_image_data(self, files_in_dir, idx):
        files_in_dir, indices = self.stride_through_frames(files_in_dir)
        data = self.read_frames(files_in_dir)
        if self.isTrain:
            labels = [NORMAL_LABEL] * len(data)
        else:
            labels = self.temporal_labels[idx][indices].tolist()
        return data, labels
    
    def read_video_data(self, files_in_dir, idx):
        data, segment_indices = self.read_videos(files_in_dir)
        if self.isTrain:
            labels = np.ones(segment_indices.shape, dtype = np.int16).tolist()
        else:
            labels = [self.temporal_labels[idx][si].tolist() for si in segment_indices]
        return data, labels
    
    def create_dataset(self):
        self.data, self.labels = list(), list()
        directory_contents = read_directory_contents(join_paths([self.parent_directory, "*"]))
        for idx, content in tqdm(enumerate(directory_contents)):
            try:
                if self.isTrain:
                    video_path = content
                    frames = video2frames(video_path, read_fps = 1)
                    data, labels = self.read(frames, idx)
                else:
                    video_directory = content
                    image_files = read_directory_contents(join_paths([video_directory, "*"]))
                    data, labels = self.read(image_files, idx)
                    
                assert len(data) == len(labels), "Data and Labels length mismatch"
            except Exception as e:
                print(e)
                continue
            if self.isTrain:
                self.data += data
                self.labels += labels
            else:
                self.data.append(data)
                self.labels.append(labels)
                    
def select_dataset(
    dataset,
    parent_path = "../../../datasets/VAD_Datasets/",
    isTrain = True,
    asImages = True,
    image_size = 128,
    image_type = "normal",
    n_frames = 16,
    frame_strides = [1,2,4,8,16],
    sample_stride = 1,
):
    if not isTrain:
        asImages = True
        sample_stride = 1
        
    kwargs = {
        "parent_path": parent_path,
        "isTrain": isTrain,
        "asImages": asImages,
        "image_size": image_size,
        "image_type": image_type,
        "n_frames": n_frames,
        "frame_strides": frame_strides,
        "sample_stride": sample_stride,
    }
    
    dataset = dataset.lower()
    flow_channels = 0
    if image_type == "flow": flow_channels += 3
    if "ucsd1" in dataset:
        return UCSD(dataset_type = 1, **kwargs), 1 + flow_channels
    elif "ucsd2" in dataset:
        return UCSD(dataset_type = 2, **kwargs), 1 + flow_channels
    elif "street" in dataset:
        return StreetScene(**kwargs), 3 + flow_channels
    elif "avenue" in dataset:
        return Avenue(**kwargs), 3 + flow_channels
    elif "shangai" in dataset:
        return ShangaiTech(**kwargs), 3 + flow_channels
    elif "subway" in dataset:
        if "entrance" in dataset: dataset_type = 0
        else: dataset_type = 1
        return Subway(dataset_type = dataset_type, **kwargs), 1 + flow_channels