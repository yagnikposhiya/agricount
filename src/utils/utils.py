"""
author: Yagnik Poshiya
github: @yagnikposhiya
organization: Tvisi
"""

import os
import json

from typing import Any

class DirectoryDoesNotExist(BaseException): # define a custom class for exception and raise if directory does not exist/available
    pass

def merge_json_files(input_files:list, output_file:str) -> Any:
    """
    This function is used to merge json files which have same file structure.

    Parameters:
    - input_files (list): list of JSON files
    - output_file (str): path to the JSON where merged JSON data will be stored

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
