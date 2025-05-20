import pandas as pd
import numpy as np
from dataclasses import dataclass
import sys
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import custom_exception
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns=['reading_score', 'writing_score']
            categorical_columns=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline=Pipeline(steps=[

                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler(with_mean=False))
            ])


            cat_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(sparse_output=False)),
                ("scaler",StandardScaler())

            ])
            logging.info("Numerical columns encoding completed")

            logging.info("Categorical columns encoding completed")

            preprocessor=ColumnTransformer([("num_pipeline",num_pipeline,numerical_columns),("cat_pipeline",cat_pipeline,categorical_columns)])

            return preprocessor
        except Exception as e:
            raise custom_exception(e,sys)
        
    def initiate_Data_Transformer(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data succefull")
            logging.info("Obtaining preprocesor object")
            preprocessiong_obj=self.get_data_transformer_object()
            target_column_name="math_score"
            numerical_columns=['reading_score', 'writing_score']
            input_feature_train_df=train_df.drop(columns=[target_column_name])
            output_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name])
            output_feature_test_df=test_df[target_column_name]

            logging.info("Data transformation initiated")

            input_feature_train_arr=preprocessiong_obj.fit_transform(input_feature_train_df)

            input_feature_test_arr=preprocessiong_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                 input_feature_train_arr,np.array(output_feature_train_df)
            ]
            test_arr=np.c_[
                 input_feature_test_arr,np.array(output_feature_test_df)
            ]

            logging.info("Saved Preprocessing objext")
            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessiong_obj)
            
            return (train_arr,test_arr,self.data_transformation_config.preprocessor_ob_file_path,)

        except Exception as e:
            raise custom_exception(e,sys)