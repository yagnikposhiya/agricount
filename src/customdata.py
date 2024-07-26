"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: Tvisi
"""

import os
import cv2
import json
import torch
import shutil
import random
import numpy as np
import pytorch_lightning as pl

from PIL import Image
from typing import Any
from collections import Counter
from torch.utils.data import Dataset, DataLoader, random_split

class FileDoesNotExist(BaseException): # create custom class to raise an error
    pass # no actions are needed

class DirectoryDoesNotExist(BaseException): # create custom class to raise an error
    pass # no actions are needed

def look_at_json_structure(json_file_path: str) -> None:
    """
    This function is used to understand the JSON file structure and what information it contains.

    Parameters:
    - json_file_path (str): Input json file path which contains mask region XY co-ordinates

    Returns:
    - (None)
    """

    input_file = open(json_file_path, 'r') # open json file in read-only mode
    input_data = json.load(input_file) # laod json data from input_file

    # print(input_data) # print json data in output console
    # print(type(input_data)) # <class 'dict'>

    print("- Keys available in json data: \n{}\n".format(input_data.keys())) # all major keys available in the json data
    # print("- Values of keys available in json data: \n{}".format(input_data.values())) # all major values available in the json data

    """
    Major keys available in json file:
    dict_keys(['_via_settings', '_via_img_metadata', '_via_attributes', '_via_data_format_version', '_via_image_id_list'])

    Out of these keys '_via_img_metadata' key contains mask region information.
    """

    print("- Keys in value of '_via_img_metadata' key: \n{}\n".format(input_data['_via_img_metadata'].keys())) # keys in value of '_via_img_metadata' key
    # as an output of above line, we'll get name of the images for those mask regions are there.

    print("- Keys in value of '9.jpeg136245' key: \n{}\n".format(input_data['_via_img_metadata']['9.jpeg136245'].keys())) # keys in value of '9.jpeg136245' key
    # dict_keys(['filename', 'size', 'regions', 'file_attributes'])

    print("- Filename: {}".format(input_data['_via_img_metadata']['9.jpeg136245']['filename'])) # filename
    # print("- Regions: {}".format(type(input_data['_via_img_metadata']['9.jpeg136245']['regions']))) # <class 'list'>
    print("- Regions: {}".format(len(input_data['_via_img_metadata']['9.jpeg136245']['regions']))) # total of regions available in an image
    print("- First region information:")
    print("-- Class name: {}".format(input_data['_via_img_metadata']['9.jpeg136245']['regions'][0]['region_attributes']['name'].rstrip('\n'))) # extract class name or extract main part of a string; do not want last \n character
    print("-- X co-ordinates: \n{}".format(input_data['_via_img_metadata']['9.jpeg136245']['regions'][0]['shape_attributes']['all_points_x'])) # X co-ordinates
    print("-- Y co-ordinates: \n{}\n".format(input_data['_via_img_metadata']['9.jpeg136245']['regions'][0]['shape_attributes']['all_points_y'])) # Y co-ordinates
        
def createMasks(json_file_path: str, raw_image_dir:str, base_data_path:str) -> str:
    """
    This function is used to create mask image from the existing information available in the 
    json file related to mask regions in each image.

    Parameters:
    - json_file_path (str): Input json file path which contains mask region XY co-ordinates
    - raw_image_dir (str): Directory path contains raw images mentioned in the json file
    - base_data_path (str): Path for data directory

    Returns:
    - (None)
    """

    input_file = open(json_file_path, 'r') # open json file in read-only mode
    input_data = json.load(input_file) # laod json data from input_fi

    if not (os.path.exists(raw_image_dir)): # check if images directory exist or not
        raise DirectoryDoesNotExist(raw_image_dir) # raise an error directory does not exist
    elif not (os.path.exists(f'{base_data_path}/processed')): # check if masks directory exists or not
        os.makedirs(f'{base_data_path}/processed') # if not then create it
    
    if not (os.path.exists(f'{base_data_path}/processed/train')): # check if train directory exists or not
        os.makedirs(f'{base_data_path}/processed/train') # if not then create it

        if not (os.path.exists(f'{base_data_path}/processed/train/images')): # check if images directory exists or not
            os.makedirs(f'{base_data_path}/processed/train/images') # if not then create it
            
        if not (os.path.exists(f'{base_data_path}/processed/train/masks')): # check if masks directory exists or not
            os.makedirs(f'{base_data_path}/processed/train/masks') # if not then create it

    trainset_path = f'{base_data_path}/processed/train' # set trainset path
    trainset_images_path = f'{base_data_path}/processed/train/images' # set trainset images path
    # input_data = input_data['_via_img_metadata'] # extract data related to mask regions only

    for filename in os.listdir(raw_image_dir): # list all files available in the images directory of raw directory
        full_filename = os.path.join(raw_image_dir,filename) # create full filename
        if os.path.exists(full_filename): # check if file available or not
            shutil.copy(full_filename,trainset_images_path) # if available then copy that file to destination path
        else:
            raise FileDoesNotExist(full_filename) # raise an error file does not exist
        
    file_names = [] # create an empty list to store filenames
    heights = [] # create an empty list to store image heights
    widths = [] # create an empty list to store image widths
    channels = [] # create en empty list to store image channels

    for key,value in input_data.items():
        filename = value ['filename'] # extract filename i.e. 9.jpeg
        image_path = f'{trainset_path}/images/{filename}' # set processed image path
        mask_path = f'{trainset_path}/masks/{filename}' # set processed mask path

        if not (os.path.exists(image_path)): # check whether image exist or not
            raise FileDoesNotExist(image_path) # raise an error file does not exist
        else:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR) # load colored image
            h, w, channel = image.shape
            
            heights.append(h) # append height of corresponding image
            widths.append(w) # append width of corresponding image
            channels.append(channel) # append channel of corresponding image
            file_names.append(image_path) # append image path

    # finding most common height and width
    most_common_height = Counter(heights).most_common(1)[0][0] # SINGLE most common height; returns a list contains tuple (common_height,count); here common_height is favorable then additional [0] index at last
    most_common_width = Counter(widths).most_common(1)[0][0] # SINGLE most common width; same explanation as above
    most_common_channel = Counter(channels).most_common(1)[0][0] # SINGLE most common channel; same explanation as above

    filtered_filenames = [] # define empty list to store filenames

    # The zip function is particularly useful when you want to iterate over multiple sequences in parallel/simultaneously.
    for filename_local, height, width, channel in zip(file_names, heights, widths, channels): # create 
        if height == most_common_height and width == most_common_width and channel == most_common_channel:
            filtered_filenames.append(filename_local)
        else:
            os.remove(filename_local) # if file has different shape than common shape then remove file from processed directory
            print(f'File has been removed from processed dataset.: {filename_local}')

    for key, value in input_data.items():
        image_path = f'{trainset_path}/images/{value['filename']}' # create full image path

        if image_path in filtered_filenames: # if image_path is available in the filtered_filenames
            image = cv2.imread(image_path, cv2.IMREAD_COLOR) # load colored image
            h, w, _ = image.shape # extract height and width of the image

            mask = np.zeros((h,w)) # create zero-array of size (height,width) for mask

            regions = value['regions'] # extract mask regions related information to the loaded image

            for region in regions:
                shape_attributes = region['shape_attributes'] # extract information regarding shape attributes
                x_points = shape_attributes['all_points_x'] # extract all x co-ordinates
                y_points = shape_attributes['all_points_y'] # extract all y co-ordinates

                contours = [] # define an empty list to store contours

                for x, y in zip(x_points,y_points):
                    contours.append((x,y)) # append tuple of x-y co-ordinates to the contours list

                contours = np.array(contours) # convert contours from list to numpy array

                cv2.drawContours(mask, [contours], -1, 255, -1) # draw contours in an image

            # apply morphological operations
            kernel = np.ones((3,3), np.uint8) # define kernel for morphological operation
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2) # used to separate touching objects

            mask = np.uint8(mask/255) # label the pixel; either 0 (background) or 1 (Peanut Seed)
            
            mask_path = f'{trainset_path}/masks/{value['filename']}' # set path for saving a masks
            cv2.imwrite(mask_path, mask) # save mask to the specified path with the same filename as image has to local storage

        else:
            print(f'File is not available in the dataset.: {image_path}') # if file(image_path) is not available in the filtered_filenames list

    print("Mask image are created successfully.")

class SegmentationDataset(Dataset):
    def __init__(self, images_dir:str, masks_dir:str, transform:bool=False) -> None:
        self.images_dir = images_dir # set path to directory contains input images
        self.masks_dir = masks_dir # set path to directory contains mask images
        self.transform = transform # set boolean value for transform/augmentation

        self.image_files = sorted(os.listdir(images_dir)) # generate a list of input images
        self.mask_files = sorted(os.listdir(masks_dir)) # generate a list of mask images

    def __len__(self) -> int:
        return len(self.image_files) # return total number of data samples available in the dataset
    
    def __getitem__(self,index) -> Any:
        image_path = os.path.join(self.images_dir, self.image_files[index]) # generate path for a single image
        mask_path = os.path.join(self.masks_dir, self.mask_files[index]) # generate path for a single mask image

        image = cv2.imread(image_path, cv2.IMREAD_COLOR) # read an input image as a colored image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # read a mask image as a grayscale image

        """
        There is no need to do anything related to one-hot encoding for mask images. Once you have clarify the number of classes
        at the time of the model implementation then it will work in the segmentation task.
        """

        """
        Tensors with negative strides are not currently supported that is why we have to use PIL and 
        then have to apply transformations.
        """
        
        image = Image.fromarray(image).convert("L") # convert to PIL grayscale images
        mask = Image.fromarray(mask).convert("L") # convert to PIL grayscale images

        if self.transform: # if transform/augmentation value is set to True
            random_int = random.randint(0,2) # return random interge including both end-points
            if random_int == 0:
                image = image # no transformation/augmentation technique is applied
            elif random_int == 1:
                image = np.flip(image, axis=0) # i.e. vertical flipping (first raw & last raw) -> (last raw -> first raw)
                mask = np.flip(mask, axis=0) # i.e. vertical flipping
            elif random_int == 2:
                image = np.flip(image, axis=1) # i.e. horizontal flipping (first column & last column) -> (last column -> first column)
                mask = np.flip(mask, axis=1) # i.e. horizontal flipping

        # convert back to numpy arrays
        image = np.array(image)
        mask = np.array(mask)

        # remove negative strides
        image = image.copy()
        mask = mask.copy()

        image = torch.tensor(image, dtype=torch.float32) # convert to torch tensor
        mask = torch.tensor(mask, dtype=torch.long) # convert to torch tensor

        return image, mask # return image and mask in the form of tuple
    
class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, train_image_dir:str, train_mask_dir:str, batch_size:int=32, transform:bool=False, val_split:float=0.2, test_split:float=0.1) -> None:
        super(SegmentationDataModule, self).__init__()
        self.train_image_dir = train_image_dir # set path to directory contains input images for training
        self.train_mask_dir = train_mask_dir # set path to directory contains input mask images for training
        self.batch_size = batch_size # set batch size
        self.transform = transform # set boolean value for transformation/augmentation
        self.val_split = val_split # set validation set split ratio
        self.test_split = test_split # set test set split ratio

    def setup(self, stage=None) -> None:
        self.train_dataset = SegmentationDataset(self.train_image_dir, self.train_mask_dir, self.transform) # create an instance of SegmentationDataset class and load data samples as needed
        # self.val_dataset = SegmentationDataset(self.val_image_dir, self.val_mask_dir) # same as trainset loading just transform is not applied to the validation set
        val_size = int(len(self.train_dataset) * self.val_split) # calculate validation set size
        # test_size = int(len(self.train_dataset) * test_size) # calculate test set size
        # train_size = len(self.train_dataset) - val_size - test_size # calculate train set size
        train_size = len(self.train_dataset) - val_size # calculate train set size

        # self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.train_dataset, [train_size,val_size, test_size]) # split whole dataset into train and validation
        self.train_dataset, self.val_dataset = random_split(self.train_dataset, [train_size,val_size]) # split whole dataset into train and validation

    def train_dataloader(self) -> Any:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True) # return train set
    
    def val_dataloader(self) -> Any:
        return DataLoader(self.val_dataset, batch_size=self.batch_size) # return validation set
    
    # def test_dataloader(self) -> Any:
    #     return DataLoader(self.test_dataset, batch_size=self.batch_size) # return test set