"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: Tvisi
"""

import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from config.config import Config
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

if __name__ == '__main__':

    config = Config() # create an instance of config class

    image = cv2.imread(config.SAMPLE_IMAGE_PATH) # load an image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert an image from BGR2RGB

    sam_checkpoint = "src/sam_vit_h_4b8939.pth" # set model checkpoint
    model_type = "vit_h" # set model type

    device = "cuda" if torch.cuda.is_available() else "cpu" # set available device
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) # load a model from model registry
    sam.to(device=device) # move model to GPU if it is available

    mask_generator_ = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.9,
        stability_score_thresh=0.96,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100, # required opencv to run post-processing
    )

    masks = mask_generator_.generate(image)
    print(len(masks))
    