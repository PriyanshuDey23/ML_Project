# Any Functionality we are writing in a common way
import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path) # take file path
        os.makedirs(dir_path,exist_ok=True) # Make Directory

        with open(file_path, 'wb') as file_obj: # Open
            pickle.dump(obj, file_obj)  # Pickle file creation, saved in the file path

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(x_train, y_train,x_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]  # Getting each and every model
            para=param[list(models.keys())[i]] # Getting all th eparams

            gs = GridSearchCV(model,para,cv=5)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_) # Select best parameters
            

            model.fit(x_train, y_train)  # Train model

            y_train_pred = model.predict(x_train) # Prediction on x train

            y_test_pred = model.predict(x_test) # Prediction on x test

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score # test report
            # report[list(models.keys())[i]] = train_model_score # Train report

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj: 
            return pickle.load(file_obj) # Load the pickle file
    except Exception as e:
        raise CustomException(e, sys)
        