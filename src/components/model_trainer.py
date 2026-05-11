import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
)
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None

from src.logger import get_logger
from src.exception import CustomException
from src.utils import evaluate_models, save_object
from dataclasses import dataclass
from sklearn.metrics import r2_score, mean_absolute_error

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl") 

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.logger = get_logger(__name__)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            self.logger.info("split training and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "decision tree": DecisionTreeRegressor(),
                "Linear Regression": linear_model.LinearRegression(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Extra Trees": ExtraTreesRegressor(),
            }
            
            if XGBRegressor is not None:
                models["XGBRegressor"] = XGBRegressor()
            if CatBoostRegressor is not None:
                models["CatBoosting Regressor"] = CatBoostRegressor(verbose=False)

            model_report: dict = evaluate_models(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, models=models)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name] 

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)    
            self.logger.info("Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            self.logger.info("Trained model saved successfully")        

            predicted = best_model.predict(x_test)
            r2_sq = r2_score(y_test, predicted)
            mae = mean_absolute_error(y_test, predicted)
            
            self.logger.info(f"Model Performance - R2 Score: {r2_sq}, MAE: {mae}")
            return best_model
     
        except Exception as e:
            self.logger.error("Error occurred in the model trainer component", exc_info=True)
            raise CustomException(e, sys)   


if __name__ == "__main__":
    print("ModelTrainer module loaded successfully")




