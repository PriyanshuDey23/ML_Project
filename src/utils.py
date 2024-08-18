# Any Functionality we are writing in a common way
import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path) # take file path
        os.makedirs(dir_path,exist_ok=True) # Make Directory

        with open(file_path, 'wb') as file_obj: # Open
            dill.dump(obj, file_obj)  # Pickle file creation, saved in the file path

    except Exception as e:
        raise CustomException(e, sys)