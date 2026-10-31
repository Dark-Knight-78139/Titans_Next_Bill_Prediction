import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import(
  AdaBoostRegressor,
  GradientBoostingRegressor,
  RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
  trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config=ModelTrainerConfig()
  
  def inititate_model_trainer(self,train_arr,test_arr):
    try:
      logging.info("Split training and test input data")
      X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
      X_test,  y_test  = test_arr[:, :-1],  test_arr[:, -1]
      models ={
       "Random Forest": RandomForestRegressor(),
       "Decision Tree": DecisionTreeRegressor(),
       "Linear Regression": LinearRegression(),
       "AdaBoost Classifier": AdaBoostRegressor(),
      }
      params={
                "Decision Tree": {
                    'criterion':['squared_error', 'absolute_error', 'poisson'],
                    
                },
                "Random Forest":{
                    
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "AdaBoost Classifier":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
      model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

      best_model_score = max(sorted(model_report.values()))

      best_model_name = list(model_report.keys())[
        list(model_report.values()).index(best_model_score)
      ]
      best_model = models[best_model_name]

      if best_model_score<0.6:
        raise CustomException("No best Model found")
      logging.info(f"Best found model on both training and testing dataset")

      save_object(
        file_path=self.model_trainer_config.trained_model_file_path,
        obj=best_model
      )

      predicted=best_model.predict(X_test)
      r2_square = r2_score(y_test,predicted)
      mae=mae(y_test,predicted)
      rmse=rmse(y_test,predicted)
      return r2_square,
    except Exception as e:
      raise CustomException(e,sys)