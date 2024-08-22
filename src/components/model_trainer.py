# Try with other model 
import os
import sys
from dataclasses import dataclass
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import ElasticNet
from scipy.stats import randint,uniform
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig: # Inputs required
    trained_model_file_path=os.path.join("artifacts","model.pkl") # save Model path

class ModelTrainer:
    def __init__(self):
        self.Model_Trainer_config=ModelTrainerConfig()

    def Initiate_Model_Trainer(self,train_array,test_array):
        try:
            logging.info("Splitting , Training and Test Input Data")
            
            # Dividing Training Dataset
            
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )            

            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boosting":GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "K-Neighbours":KNeighborsRegressor(),
                "Adaboost":AdaBoostRegressor(),
                "ElasticNet":ElasticNet()
            }

            

         # Hyper Parameter Tuning
            params = {
                "Random Forest": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]
                },
                "Decision Tree": {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['mse', 'friedman_mse', 'mae']
                },
                "Gradient Boosting": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                },

                "Linear Regression":{},

                "K-Neighbours": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "Adaboost": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0],
                    'loss': ['linear', 'square', 'exponential']
                },
                "ElasticNet": {
                    'alpha': [0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.5, 0.9]
                }
            }

            # Evaluate model from utils
            model_report:dict=evaluate_models(                     
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                param=params
            ) 


            

            # To get best model score from dictionary
            best_model_score=max(sorted(model_report.values())) # Sort model score based on values

            # To Get the best model name from dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]  # Key is the model name
            
            
            best_model=models[best_model_name] # name

            # Threshhold
            if best_model_score<0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best found model on both training and testing dataset")

            

            save_object(
                file_path=self.Model_Trainer_config.trained_model_file_path,
                obj=best_model

            )

            predicted=best_model.predict(x_test)

            # Accuracy
            r2_square=r2_score(y_test,predicted)

            return r2_square


        except Exception as e:
            raise CustomException(e,sys)



