"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: Tvisi
"""

import glob

from config.config import Config
from gpu_config.check import check_gpu_config
from customdata import look_at_json_structure, createMasks
from utils.utils import merge_json_files


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
    output_file = f'{config.INPUT_JSON_FILE_PATH}/merge.json' # define path for json file, contains merged data from multiple json files
    merge_json_files(input_files=input_files, output_file=output_file) # merge multiple json files

    print("Creating mask images from json data...")
    createMasks(output_file, config.RAW_IMAGE_DIR, config.BASE_DATA_PATH) # create mask images from existing mask region information i.e. XY co-ordinates