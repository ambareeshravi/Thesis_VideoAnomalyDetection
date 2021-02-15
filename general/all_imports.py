import matplotlib
matplotlib.use('Agg')
import os
import shutil
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import pickle as pkl
import json
import torch
import gc

from datetime import datetime, timedelta
from time import time
from tqdm import tqdm
from PIL import Image, ImageOps

from torch import nn
from torch.nn import functional as F
from glob import glob
from matplotlib import pyplot as plt

from sklearn.metrics import *

from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from .utils import *
import cv2

from collections import OrderedDict
import random

MANUAL_SEED = 42
np.random.seed(MANUAL_SEED)
torch.manual_seed(MANUAL_SEED)
torch.cuda.manual_seed(MANUAL_SEED)
torch.cuda.manual_seed_all(MANUAL_SEED)
random.seed(MANUAL_SEED)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False