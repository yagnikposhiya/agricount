"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: NEUROWORK Research Labs
"""

import os

class Config():
    def __init__(self) -> None:

        # current working directory
        self.CWD = os.getcwd() # get current working directory

        # training and validation set paths
        self.TRAINSET_PATH = '' # set training set path
        self.VALIDSET_PATH = '' # set validation set path
        self.SAMPLE_IMAGE_PATH = os.path.join(self.CWD,'data/raw/train/1.jpeg') # set path for any images available in the trainset/train directory

        # weight and biases config
        self.ENTITY = '' # set team/organization name for wandb account
        self.PROJECT = '' # set project name
        self.REINIT = True # set boolean value for reinitialization
        self.ANONYMOUS = 'allow' # set anonymous value type
        self.LOG_MODEL = 'all' # set log model type

        # model training parameters
        self.BATCH_SIZE = 16 # set batch size for model training
        self.MAX_EPOCHS = 2 # set maximum epochs for model training
        self.NUM_CLASSES = 4 # set number of classes contains by mask images (in segmentation case)
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

        print("Configurations:")
