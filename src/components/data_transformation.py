import logging
import sys
from dataclasses import dataclass
import os
import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import get_logger
from sklearn.impute import SimpleImputer
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.logger = get_logger(__name__)

    def get_data_transformer_object(self):
        try:
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            self.logger.info("Numerical and categorical pipelines created successfully")

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_features),
                ('cat_pipeline', cat_pipeline, categorical_features)
            ])
            self.logger.info("Data transformation object created successfully")
            return preprocessor

        except Exception as e:
            self.logger.error("Error occurred in get_data_transformer_object method", exc_info=True)
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            self.logger.info("Data transformation initiated")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            self.logger.info("Read train and test data completed")


            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            self.logger.info("Applying preprocessing object on training and testing datasets")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            self.logger.info("Saved preprocessing object")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            self.logger.error("Error occurred in the initiate_data_transformation method", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    print("DataTransformation module loaded successfully")