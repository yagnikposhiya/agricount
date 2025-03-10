"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: Tvisi
"""

import os

class Config():
    def __init__(self) -> None:

        # current working directory
        self.CWD = os.getcwd() # get current working directory

        # training and validation set paths
        self.TRAINSET_PATH = '' # set training set path
        self.VALIDSET_PATH = '' # set validation set path
        self.TRAIN_IMAGE_DIR = '' # set train/images directory path
        self.TRAIN_MASK_DIR = '' # set train/masks directory path

        # other relevant paths
        self.INPUT_JSON_FILE_PATH = os.path.join(self.CWD,'data/raw/json_projects') # set path to the directory which contains one or more than one json files
        self.SAMPLE_JSON_FILE_PATH = os.path.join(self.CWD,'data/raw/json_projects/File1.json') # set path to the json file for understanding json file structure
        self.RAW_IMAGE_DIR = os.path.join(self.CWD,'data/raw/images') # set raw image directory
        self.BASE_DATA_PATH = os.path.join(self.CWD,'data') # set base data folder path
        self.PATH_TO_SAVE_TRAINED_MODEL = os.path.join(self.CWD,'saved_models') # set path to save trained model

        # weight and biases config
        self.ENTITY = 'neuralninjas' # set team/organization name for wandb account
        self.PROJECT = 'tvisi-agri-count' # set project name
        self.REINIT = True # set boolean value for reinitialization
        self.ANONYMOUS = 'allow' # set anonymous value type
        self.LOG_MODEL = 'all' # set log model type

        # model training parameters
        self.BATCH_SIZE = 16 # set batch size for model training
        self.MAX_EPOCHS = 2 # set maximum epochs for model training
        self.NUM_CLASSES = 2 # set number of classes contains by mask images (in segmentation case)
        self.LEARNING_RATE = 0.001 # set learning rate
        self.TRANSFORM = True # set booelan values for applying augmentation techniques for training set


    def printConfiguration(self) -> None:
        """
        This function is used to print all configuration related to paths and model training params

        Parameters:
        - (None)

        Returns:
        - (None)
        """

        print("-----------------------------------------------------------")
        print("-----------------------CONFIGURATIONS----------------------")
        print("-----------------------------------------------------------")
        print("\n",
              f"- Current working directory: {self.CWD}\n",
              f"- Trainset path: {self.TRAINSET_PATH}\n",
              f"- Validset path: {self.VALIDSET_PATH}\n",
              f"- Train image directory: {self.TRAIN_IMAGE_DIR}\n",
              f"- Train mask directory: {self.TRAIN_MASK_DIR}\n",
              f"- Input JSON file path: {self.INPUT_JSON_FILE_PATH}\n",
              f"- Sample JSON file path: {self.SAMPLE_JSON_FILE_PATH}\n",
              f"- Raw image directory: {self.RAW_IMAGE_DIR}\n",
              f"- Base data path: {self.BASE_DATA_PATH}\n",
              f"- Path to save trained model: {self.PATH_TO_SAVE_TRAINED_MODEL}\n",
              f"- Batch size: {self.BATCH_SIZE}\n",
              f"- Maximum epochs: {self.MAX_EPOCHS}\n",
              f"- Number of classes: {self.NUM_CLASSES}\n",
              f"- Learning rate: {self.LEARNING_RATE}\n",
              f"- Tranformation/Augmentation: {self.TRANSFORM}\n")
