import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
import os 

@dataclass
class DataTransformationConfig:
  preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
  def __init__(self):
    self.data_transformation_config=DataTransformationConfig()

  def get_data_transformer_object(self):
    '''
     This function is responsible for data transformation
    '''
    try:
      numerical_columns =[
      'Monthly rental',
      'is_unlimited_data',
      'is_unlimited_calls',
      'Extra data',
      'Extra Call min',
      'est_isd_total_cost',
      'data_utilization_ratio',
      'call_utilization_ratio',
      'Current Bill Amount',
      'prev_bill_t1',
      'bill_mom_change',
      'rolling_3m_bill_mean',
      'rolling_3m_data_mean',
      'expanding_cust_bill_mean',
      'expanding_cust_bill_std',
      ]
      num_pipeline= Pipeline(
        steps=[
          ("imputer",SimpleImputer(strategy="median")),
          ("scaler",StandardScaler())
        ]
      )
      logging.info(f"Numerical Column: {numerical_columns}")

      preprocessor=ColumnTransformer(
        [
          ("num_pipline",num_pipeline,numerical_columns),
        ]
      )
      return preprocessor
    except Exception as e:
      raise CustomException(e,sys)
  
  def initiate_data_transformation(self,train_path,test_path):
    try:
      target_column_name="target_next_bill"
      train_df = pd.read_csv(train_path).dropna(subset=[target_column_name])
      test_df = pd.read_csv(test_path).dropna(subset=[target_column_name])
      logging.info("Read train and test data completed")
      logging.info("Obtaining preprocessing object")

      preprocessing_obj=self.get_data_transformer_object()

      
      numerical_columns =['Monthly rental',
         'is_unlimited_data',
         'is_unlimited_calls',
         'Extra data',
         'Extra Call min',
         'est_isd_total_cost',
         'data_utilization_ratio',
         'call_utilization_ratio',
         'Current Bill Amount',
         'prev_bill_t1',
         'bill_mom_change',
         'rolling_3m_bill_mean',
         'rolling_3m_data_mean',
         'expanding_cust_bill_mean',
         'expanding_cust_bill_std',
      
      ]

      input_feature_train_df=train_df.drop(columns=[target_column_name])
      target_feature_train_df=train_df[target_column_name]

      input_feature_test_df=test_df.drop(columns=[target_column_name])
      target_feature_test_df=test_df[target_column_name]

      logging.info( 
        f"Applying preprocessing object on training dataframe and testing dataframe."
      )
      input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
      input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

      train_arr=np.c_[
        input_feature_train_arr, np.array(target_feature_train_df)
      ]
      test_arr=np.c_[
        input_feature_test_arr,np.array(target_feature_test_df)
      ]
      logging.info(f"Saved preprocessing object.")

      save_object(
        file_path=self.data_transformation_config.preprocessor_obj_file_path,
        obj=preprocessing_obj
      )

      return(
        train_arr,
        test_arr,
        self.data_transformation_config.preprocessor_obj_file_path,
      )
    except Exception as e:
      raise CustomException(e,sys)