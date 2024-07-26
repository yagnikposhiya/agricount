"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: Tvisi
"""

import os
import cv2
import glob
import json
import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from typing import Any

matplotlib.use('TkAgg') # or 'Qt5Agg' or any other backend that supports interactive display
# by default backend: FigureCanvasAgg

class DirectoryDoesNotExist(BaseException): # define a custom class for exception and raise if directory does not exist/available
    pass

def merge_json_files(input_files:list, output_file:str) -> Any:
    """
    This function is used to merge json files which have same file structure.

    Parameters:
    - input_files (list): List of JSON files
    - output_file (str): Path to the JSON where merged JSON data will be stored

    Returns:
    - (None)
    """

    if len(input_files) == 1:
        print(f'Only {len(input_files)} json file is detected.')
    else:
        print(f'Merging {len(input_files)} json files.')

    dictionaries_list = [] # define empty dictionaries list

    for file in input_files:
        input_file = open(file, 'r') # open json file in read-only mode
        input_data = json.load(input_file) # load json data from input_file
        input_data = input_data['_via_img_metadata'] # extract values of _via_img_metadata key
        dictionaries_list.append(input_data) # append extracted values to the list of dictionaries

    merged_data = {key: value for d in dictionaries_list for key, value in d.items()} # merging all dictionaries data

    directory_path, filename = os.path.split(output_file) # split output_file path into two parts; 1) directory path and 2) filename

    if os.path.exists(directory_path): # check that directory exists or not
        outfile = open(output_file, 'w') # open json file in writing mode
        json.dump(merged_data, outfile) # write merged json data to the output_file with indent=4 
    else:
        raise DirectoryDoesNotExist(directory_path) # raise an exception that directory does not exist
    
    if len(input_files) > 1: # if more than one json files are available
        print(f'{len(input_files)} json files are merged successfully.')

def show_mask_image_pair(image_dir:str, mask_dir:str) -> None:
    """
    This function is used to visualize image-mask pair.

    Parameters:
    - image_dir (str): Path to image directory in processed data directory
    - mask_dir (str): Path to mask directory in processed data directory

    Returns:
    - (None)
    """

    image_list = glob.glob(f'{image_dir}/*.jpeg') # create list contains all images
    mask_list = glob.glob(f'{mask_dir}/*.jpeg') # create list contains all masks

    while True:
        try:
            user_choice = str(input('Do you want to visualize mask-image pairs? [Y/N]: ')) # ask user for his/her binary choice
            if user_choice.lower() == 'y': # if user enter Y/y
                random_num = random.randint(0,len(image_list)-1) # generate random number between 0 and len(image_list)-1
                image = cv2.imread(image_list[random_num], cv2.IMREAD_COLOR) # read an image
                mask = cv2.imread(mask_list[random_num], cv2.IMREAD_GRAYSCALE) # read a mask

                _, image_name = os.path.split(image_list[random_num]) # extract image name from the path
                _, mask_name = os.path.split(mask_list[random_num]) # extract mask name from the path

                # opacity = 0.5 # set desired opacity for the mask image
                # mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # convert mask to a 3-channel image

                # # apply a color to a mask for better visualization
                # mask_color[:, :, 1] = 0 # zero out the green channel
                # mask_color[:, :, 2] = 255 # max out the red channel

                # overlay_image = cv2.addWeighted(image, 1-opacity, mask_color, opacity, 0) # overlay the mask onto original image

                fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(40,40)) # create a canvas with 1 rows and 2 cols with 20*20 figure size

                ax1.imshow(image, cmap='gray') # visualize an image
                ax1.set_title(f'{image_name}') # set image name as a title
                ax1.axis('off') # do not visualize an image with an axis

                ax2.imshow(mask, cmap='gray') # visualize a mask
                ax2.set_title(f'{mask_name}') # set mask name as a title
                ax2.axis('off') # do not visualize an image with an axis

                # ax3.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)) # visualize an overlay image
                # ax3.set_title('Overlay image') # set title for figure
                # ax3.axis('off') # do not visualize overlay image with an axis

                plt.show() # display all plots/graphs

            elif user_choice.lower() == 'n': # if user enters either N or n
                return None # stop loop execution

            else:
                print('Enter \'Y\' or \'N\'') # ask user to enter either 'Y' or 'N'

        except BaseException:
            pass # do nothing ignore the exception

def get_user_choice(start:int, end:int) -> int:
    """
    This function is used to get integer user choice within specific range including both end-points.

    Parameters:
    - start (int): Starting point of the range
    - end (int: Ending point of the range

    Returns:
    - (int): Integer user choice
    """

    while True:
        try:
            user_choice = int(input(f'Enter an integer number between {start} and {end}: ')) # ask user to enter his/her choice

            if start <= user_choice <= end:
                return user_choice
            else:
                print(f'Invalid number. Enter an integer number between {start} and {end}') # ask user to enter a choice between specified range
        except ValueError:
            print(f'Invalid number. Enter an integer number between {start} and {end}') # ask user to enter a choice between specified range

def available_models() -> tuple:
    """
    This is used to provide a list of neural net architectures available for training on the existing dataset(s).

    Parameters:
    - (None)

    Returns:
    - (list,int): Returns tuple contains list of available neural net archs and user choice; (available_nn_arch, user_choice)
    """

    models = ['DNA-Segment'] # list of available models

    print('Select any one neural net architecture from the list given below')
    for i in range(len(models)):
        print(f'{i}_________{models[i]}') # print list of available models with the integer model number

    if (len(models)-1) == 0:
        return models, np.uint8(0) # only one neural net architecture is there; no need to ask to user for their choice
    else:
        return models, get_user_choice(0,len(models)-1) # get user choice
    
def available_optimizers() -> tuple:
    """
    This function is used to provide list of optimizers available for selected neural net architectures.

    Parameters:
    - (None)

    Returns:
    - (list,int): Returns tuple contains list of available optimizers and user choice; (available optimizers, user_choice)
    """

    optimizers = ['Adam',
                  'AdamW',
                  'RMSProp',
                  'SGD'] # list of available optimizers
    
    print('Select any one optimizer from the list given below')
    for i in range(len(optimizers)):
        print(f'{i}_________{optimizers[i]}') # print list of available optimizers with the integer optimizer number

    if len(optimizers) == 0:
        return optimizers, np.uint(0) # only one optimizer is there; no need to ask to user for their choice
    else:
        return optimizers, get_user_choice(0, len(optimizers)-1) # get user choice