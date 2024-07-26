"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: Tvisi
"""

import os
import glob
import torch

from config.config import Config
from gpu_config.check import check_gpu_config
from customdata import SegmentationDataModule
from customdata import look_at_json_structure, createMasks
from utils.utils import merge_json_files, show_mask_image_pair


if __name__ == '__main__':

    check_gpu_config() # get GPU (General Processing Unit) information if it is available

    config = Config() # create an instance of Config class
    config.printConfiguration() # print all configuration set by defualt

    print("-----------------------------------------------------------")
    print("--------------UNDERSTANDING JSON FILE STRUCTURE------------")
    print("-----------------------------------------------------------")

    look_at_json_structure(config.SAMPLE_JSON_FILE_PATH) # understand JSON file structure and which information it contains

    print("-----------------------------------------------------------")
    print("------------------CREATING PROCESSED DATASET---------------")
    print("-----------------------------------------------------------")
    
    input_files = glob.glob(f'{config.INPUT_JSON_FILE_PATH}/File*.json') # create list of json files available @ config.INPUT_JSON_FILE_PATH
    output_file = f'{config.INPUT_JSON_FILE_PATH}/Merge.json' # define path for json file, contains merged data from multiple json files
    merge_json_files(input_files=input_files, output_file=output_file) # merge multiple json files

    print("Creating mask images from json data...")
    createMasks(output_file, config.RAW_IMAGE_DIR, config.BASE_DATA_PATH) # create mask images from existing mask region information i.e. XY co-ordinates

    config.TRAINSET_PATH = os.path.join(config.BASE_DATA_PATH,f'processed/train')

    show_mask_image_pair(image_dir=os.path.join(config.TRAINSET_PATH,'images'),
                         mask_dir=os.path.join(config.TRAINSET_PATH,'masks')) # visualize mask-image pairs
    
    config.TRAIN_IMAGE_DIR = os.path.join(config.TRAINSET_PATH,'images') # set path to a directory contains input images for training
    config.TRAIN_MASK_DIR = os.path.join(config.TRAINSET_PATH, 'masks') # set path to a directory contains input mask images for training

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # set device for model training
    data_module = SegmentationDataModule(train_image_dir=config.TRAIN_IMAGE_DIR,
                                         train_mask_dir=config.TRAIN_MASK_DIR,
                                         batch_size=config.BATCH_SIZE,
                                         transform=config.TRANSFORM) # initialize the data module