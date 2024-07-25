"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: Tvisi
"""

from config.config import Config
from gpu_config.check import check_gpu_config
from customdata import look_at_json_structure, createMasks


if __name__ == '__main__':

    check_gpu_config() # get GPU (General Processing Unit) information if it is available

    config = Config() # create an instance of Config class
    config.printConfiguration() # print all configuration set by defualt

    print("-----------------------------------------------------------")
    print("--------------UNDERSTANDING JSON FILE STRUCTURE------------")
    print("-----------------------------------------------------------")

    look_at_json_structure(config.INPUT_JSON_FILE_PATH) # understand JSON file structure and which information it contains

    print("-----------------------------------------------------------")
    print("------------------CREATING PROCESSED DATASET---------------")
    print("-----------------------------------------------------------")
    
    print("Creating mask images from json data...")
    createMasks(config.INPUT_JSON_FILE_PATH, config.RAW_IMAGE_DIR, config.BASE_DATA_PATH) # create mask images from existing mask region information i.e. XY co-ordinates