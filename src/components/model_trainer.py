import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor,)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


from src.exception import custom_exception
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiats_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("split train and test input data")
            x_train,y_train,x_test,y_test=(train_arr[:,:-1],train_arr[:,-1],test_arr[:,:-1],test_arr[:,-1])

            models={        "Random Forest":RandomForestRegressor(),
                            "Gradient Boosting":GradientBoostingRegressor(),    
                            "KNN":KNeighborsRegressor(),
                            "LinearRegression":LinearRegression(),
                            "DecisionTreeRegressor":DecisionTreeRegressor(),
                            "CatBoostRegressor":CatBoostRegressor(verbose=False),
                            "AdaBoostRegressor":AdaBoostRegressor(),
                            "XGBoost":XGBRegressor()      
                    }
            
            hyperparams={
                            "Random Forest": {
                                "n_estimators": [100, 200, 500],
                                "max_depth": [None, 10, 20, 30],
                                "min_samples_split": [2, 5, 10],
                                "min_samples_leaf": [1, 2, 4],
                                "max_features": ["auto", "sqrt", "log2"]
                            },
                            "Gradient Boosting": {
                                "n_estimators": [100, 200],
                                "learning_rate": [0.01, 0.1, 0.2],
                                "max_depth": [3, 5, 10],
                                "min_samples_split": [2, 5, 10],
                                "min_samples_leaf": [1, 2, 4],
                                "subsample": [0.6, 0.8, 1.0],
                                "max_features": ["auto", "sqrt", "log2"]
                            },
                            "KNN": {
                                "n_neighbors": [3, 5, 10, 15],
                                "weights": ["uniform", "distance"],
                                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                                "p": [1, 2]
                            },
                            "LinearRegression": {
                                "fit_intercept": [True, False],
                               
                            },
                            "DecisionTreeRegressor": {
                                "max_depth": [None, 10, 20, 30],
                                "min_samples_split": [2, 5, 10],
                                "min_samples_leaf": [1, 2, 4],
                                "max_features": ["auto", "sqrt", "log2"]
                            },
                            "CatBoostRegressor": {
                                "iterations": [100, 200, 500],
                                "learning_rate": [0.01, 0.05, 0.1],
                                "depth": [4, 6, 10],
                                "l2_leaf_reg": [1, 3, 5, 7],
                                "bootstrap_type": ["Bayesian", "Bernoulli", "MVS"]
                            },
                            "AdaBoostRegressor": {
                                "n_estimators": [50, 100, 200],
                                "learning_rate": [0.01, 0.1, 1.0],
                                "loss": ["linear", "square", "exponential"]
                            },
                            "XGBoost": {
                                "n_estimators": [100, 200, 500],
                                "learning_rate": [0.01, 0.1, 0.2],
                                "max_depth": [3, 6, 10],
                                "subsample": [0.6, 0.8, 1.0],
                                "colsample_bytree": [0.6, 0.8, 1.0],
                                "gamma": [0, 1, 5],
                                "reg_alpha": [0, 0.1, 1],
                                "reg_lambda": [1, 1.5, 2]
                            }
                            }

            
            model_report:dict=evaluate_models(X_train=x_train,y_train=y_train,X_test=x_test,y_test=y_test,models=models,hyperparams=hyperparams)

            best_model_score=max(sorted(model_report.values()))
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]
            print(best_model_name)
            if best_model_score < 0.6:
                raise custom_exception.TrainingException("Model score is less than 0.6")
            logging.info("Best found model loaded")
            save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=best_model)

            predicted=best_model.predict(x_test)
            r2_square=r2_score(y_test,predicted)
            return r2_square
            
        except Exception as e:
            raise custom_exception(e,sys)
