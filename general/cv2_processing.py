import numpy as np
np.random.seed(0)

import cv2

class OpticalFlow:
    def __init__(self, returnMag = True):
        self.returnMag = returnMag
    
    def convert_to_gray(self, image):
        if image.shape[-1]!=3: return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def get_optical_flow(self, frame1, frame2):
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        p_frame = self.convert_to_gray(frame1)
        n_frame = self.convert_to_gray(frame2)
        flow = cv2.calcOpticalFlowFarneback(p_frame, n_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        optical_flow_mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        if self.returnMag:
            return np.uint8(optical_flow_mag) # returns 1 channel magnitude
        
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = optical_flow_mag
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return rgb_flow # returns 3 channel flow
    
class BackgroundSubtraction:
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        
    def __call__(self, frame):
        fgmask = self.fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
        return fgmask