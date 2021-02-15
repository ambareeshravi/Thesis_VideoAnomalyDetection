import sys
sys.path.append("..")
from general import *
import cv2

# utils
def threshold_image(image, amount = 0.9):
#     return image_255(image > (np.max(image)*amount))
    return image_255(image > np.percentile(image, 100*amount))

def erode_image(image, e_kernel = np.ones((2,2),np.uint8), iterations = 1):
    return cv2.erode(image, e_kernel, iterations = iterations)

def morph_open_image(image, m_kernel = np.ones((3,3),np.uint8)):
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, m_kernel)

def get_difference(x, y):
    return np.abs(array_to_gray(x)-array_to_gray(y))

def mask_image(original, mask, color = [255,0,0]):
    base = array_3channels(original).copy()
    x_coord, y_coord = np.where(mask > 0)
    for x,y in zip(x_coord, y_coord):
        base[x, y] = color
    return base

# vis functions
def check_viz_input(x):
    assert len(x.shape) == 3, "Dimension for image should be 3"
    assert isinstance(x, np.ndarray), "Type np.array expected"
    
    if x.shape[-1] == 1: x = x.squeeze(axis = -1)
    return x
    
def visualize_anomalies(original, reconstruction, difference_map):
    original = check_viz_input(image_255(original)) # w,h,c [0,255]
    reconstruction = check_viz_input(image_255(reconstruction)) # w,h,c [0,255]
    
    diff = image_255(difference_map) # diff should always be grayscale w,h [0,255]
    thresh = threshold_image(diff) # w,h [0,255]
    
    erosion = erode_image(thresh) #w,h [0,255]
    morph = morph_open_image(erosion) #w,h [0,255]
    masked = mask_image(original, morph) #w,h,3
    
    result_images = [original, reconstruction, diff, erosion, thresh, morph, masked]
    return list(map(array_3channels, result_images))

def frames_to_video(frames_list, video_path, ext = ".mp4", fps = 30, filpChannels = True):
    video_path += ext
    h,w,c = frames_list[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_path, fourcc, float(fps), (w,h))
    for frame in frames_list:
        if filpChannels: frame = frame[:,:,::-1]
        video.write(frame)
    video.release()