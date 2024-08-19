import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_object_file_path= os.path.join('artifacts',"preprocessor.pkl")   # For saving the model in pickle file

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    # To create all my pickle files , which will be responsible for converting category and numerical
    def get_data_transformer_object(self): 
        """
            This Function is responsible for Data transformation
        """
        try:
            numerical_columns=["reading_score","writing_score"]
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # For Handling the missing values
            num_pipeline=Pipeline(
                steps=[
                ("Imputer",SimpleImputer(strategy="median")), # median
                ("Scaler",StandardScaler())
                ]
            )

            logging.info("Numerical columns Standard Scaling Completed")

            cat_pipeline=Pipeline(
                steps=[
                ("Imputer",SimpleImputer(strategy="most_frequent")), # Mode
                ("One_Hot_Encoder",OneHotEncoder(sparse_output=True)), # Default is sparse # sparse_output=True: The encoder outputs a sparse matrix. Sparse matrices are memory-efficient representations of large matrices with a lot of zeros. 
                ("Scaler",StandardScaler(with_mean=False)) # Prevent centering #with_mean=False: This prevents the StandardScaler from subtracting the mean from each feature.
                ]
            )

            logging.info("Categorical columns encoding Completed")

            
            # Combine both the pipeline

            preprocessor=ColumnTransformer(
                [
                ("Numerical_Pipeline",num_pipeline,numerical_columns),
                ("Categorical_Pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)


# Starting my data tranformation

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read Train And Test Data Completed ")

            logging.info("Obtaining Preprocessing Object")

            preprocessing_object=self.get_data_transformer_object()

            target_column_name="math_score"
            

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info( f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr=preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_object.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)] #np.c_ -> concat
            
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            # saving the model
            # Check utils.py
            save_object(

                file_path=self.data_transformation_config.preprocessor_object_file_path,
                obj=preprocessing_object

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_object_file_path,
            )

           

        except Exception as e:
            raise CustomException(e,sys)
        



# run this to do the data transformation

# if __name__=="__main__":
#     object=DataTransformation()
#     object.get_data_transformer_object()