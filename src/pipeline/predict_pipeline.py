import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features): # Model prediction file
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model=load_object(file_path=model_path) # Load the model(Load the pickle file)
            preprocessor=load_object(file_path=preprocessor_path) # Load the preprocessor
            data_scaled=preprocessor.transform(features) # Scale the data
            prediction=model.predict(data_scaled)  # Prediction
            return prediction
        except Exception as e:
            raise CustomException(e, sys)
    
        

class CustomData: # It will be responsible for mapping all the inputs with the html
    def __init__( self,
        gender : str,
        race_ethnicity : str,
        parental_level_of_education: str,
        lunch : str,
        test_preparation_course : str,
        reading_score : int,
        writing_score : int):

        # Assigning the values
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self): # Create Dictionary # Return all the input in the form of data frame
        try:
            Custom_data_input_dict= {
                "gender" : [self.gender],
                "race_ethnicity" : [self.race_ethnicity],
                "parental_level_of_education" : [self.parental_level_of_education],
                "lunch" : [self.lunch],
                "test_preparation_course" : [self.test_preparation_course],
                "reading_score" : [self.reading_score],
                "writing_score" : [self.writing_score],

            }

            return pd.DataFrame(Custom_data_input_dict) # Get data frame

        except Exception as e:
            raise CustomException(e,sys)
        