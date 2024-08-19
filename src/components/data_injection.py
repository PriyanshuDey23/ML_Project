import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass  # Create class variable
from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainer



@dataclass   # Create class variables
class DataIngestionConfig: # Where I will save the raw,train,test data (Providing input thing for data injection component)
    train_data_path: str=os.path.join('artifacts','train.csv') # Input # In the artifact folder , all the outputs will be stored
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','raw.csv')

class DataIngestion: # Input from data ingest config
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()  # It will consist the train,test,raw value


    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv("notebook\data\student.csv") # Reading
            logging.info("Read the data set as Data Frame") # Log file for above

            # Create the artifacts folder and also the files
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # getting the directory name with specific path

            # For raw data
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)   # Raw and save it in artifact location
            logging.info("Train Test split initiated") # Log file for above  
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            
            # For train Data
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) 

            # For test Data
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True) 

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )


        except Exception as e:
            raise CustomException(e,sys)
        
# After Running this folder and files will be created
        
# if __name__=="__main__":
#     object=DataIngestion()
#     object.initiate_data_ingestion()




# Starting the data ingestion( checking), after data transformation

if __name__=="__main__":
    object=DataIngestion()
    train_data,test_data=object.initiate_data_ingestion()
    data_transformation=DataTransformation()
    #data_transformation.initiate_data_transformation(train_data,test_data)

# Pickle file will be generated in the artifacts

    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
    model_trainer=ModelTrainer()
    print(model_trainer.Initiate_Model_Trainer(train_arr,test_arr))

# Model .pkl file is created

