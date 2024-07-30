"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: Tvisi
"""

import os
import glob
import torch
import wandb
import pytorch_lightning as pl

from torchinfo import summary
from config.config import Config
from gpu_config.check import check_gpu_config
from customdata import SegmentationDataModule
from nn_arch.dna_segment import DNASegmentModel
from pytorch_lightning.loggers import WandbLogger
from customdata import look_at_json_structure, createMasks
from utils.utils import merge_json_files, show_mask_image_pair, available_models, available_optimizers, save_trained_model


if __name__ == '__main__':

    check_gpu_config() # get GPU (General Processing Unit) information if it is available

    config = Config() # create an instance of Config class
    config.printConfiguration() # print all configuration set by defualt

    wandb.init(entity=config.ENTITY, # assign team/organization name
               project=config.PROJECT, # assign project name
               anonymous=config.ANONYMOUS, # set anonymous value type
               reinit=config.REINIT) # initialize the weights and biases cloud server instance

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
    createMasks(json_file_path=output_file, 
                raw_image_dir=config.RAW_IMAGE_DIR, 
                base_data_path=config.BASE_DATA_PATH) # create mask images from existing mask region information i.e. XY co-ordinates

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
    
    print("-----------------------------------------------------------")
    print("---------------NN ARCHITECTURE (MODEL) SELECTION-----------")
    print("-----------------------------------------------------------")

    available_nn_archs, user_choice_nn_arch = available_models() # give list of available neural net architectures to user for training
    available_optims, user_choice_optimizer = available_optimizers() # give list of available optimizers to user for neural net configuration
    print(f'{available_nn_archs[user_choice_nn_arch]} neural net architecture is selected with {available_optims[user_choice_optimizer]} optimizer.')

    if user_choice_nn_arch == 0:
        model = DNASegmentModel(
            img_height=720, # image height
            img_width=1280, # image width
            patch_size=16, # single patch size
            in_channels=3, # number of channels in the input images
            embed_dim=768, # embedding dimension
            num_layers=18, # number of layers in the transformers
            num_heads=12, # number of heads
            mlp_dim=3072, # MLP dimension
            num_classes=2, # total number of output classes
            optimizer=available_optims[user_choice_optimizer]) # set optimizer

    print('- Model summary:\n')
    summary(model=model,
            input_size=(1,3,720,1280),
            col_names=['input_size','output_size','kernel_size']) # print model summary; input shape is extracted @ data loading time
    
    model = model.to(device) # move neural net architecture to available computing device
    wandb_logger = WandbLogger(log_model=config.LOG_MODEL) # initialize the weights-and-biases logger

    trainer = pl.Trainer(max_epochs=config.MAX_EPOCHS, # set maximum number of epochs
                         log_every_n_steps=1, # after how many 'n' steps log will be saved
                         logger=wandb_logger) # assign logger for saving a logs
    
    print("-----------------------------------------------------------")
    print("---------------NN ARCHITECTURE (MODEL) TRAINING------------")
    print("-----------------------------------------------------------")

    print('Training started...')
    trainer.fit(model, data_module) # train the neural network architecture selected by user
    print('Training finished.')

    wandb.finish() # close the weights and biases cloud instance

    print("-----------------------------------------------------------")
    print("----------------------SAVE TRAINED MODEL-------------------")
    print("-----------------------------------------------------------")

    print('Saving trained model..')
    save_trained_model(model=model, # model
                       path=config.PATH_TO_SAVE_TRAINED_MODEL, # path to save trained model
                       model_prefix=available_nn_archs[user_choice_nn_arch], # model name
                       optimizer=available_optims[user_choice_optimizer], # selected optimizer and max. epochs
                       epochs=config.MAX_EPOCHS) # save trained neural network architecture in the .pth format
