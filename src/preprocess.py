import configparser
import os
import pandas as pd
import sys
import traceback
import pickle

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from logger import Logger

SHOW_LOG = True

class CatBoostClassifierTrainer:
    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        

        try:
            split_data = self.config["SPLIT_DATA"]
            self.X_train_path = split_data["X_train"]
            self.y_train_path = split_data["y_train"]
            self.X_test_path = split_data["X_test"]
            self.y_test_path = split_data["y_test"]
        except KeyError as e:
            self.log.error("В config.ini отсутствует секция SPLIT_DATA или нужные ключи: " + str(e))
            sys.exit(1)
        
        try:
            self.X_train = pd.read_csv(self.X_train_path, index_col=0)
            self.y_train = pd.read_csv(self.y_train_path, index_col=0)
            self.X_test = pd.read_csv(self.X_test_path, index_col=0)
            self.y_test = pd.read_csv(self.y_test_path, index_col=0)
        except Exception as e:
            self.log.error("Ошибка при чтении данных: " + str(e))
            sys.exit(1)
        
        # Сохраняем модель
        self.project_path = os.path.join(os.getcwd(), "experiments")
        if not os.path.exists(self.project_path):
            os.makedirs(self.project_path)
        self.model_path = os.path.join(self.project_path, "catboost_model.sav")
        self.log.info("CatBoostClassifierTrainer is ready")

    def train_model(self, use_config: bool = False, iterations: int = 1000, depth: int = 6, learning_rate: float = 0.1, predict: bool = False) -> bool:
        """
        Обучает модель CatBoostClassifier.
        
        Если use_config=True, параметры берутся из секции [CATBOOST] файла config.ini.
        Если predict=True, выводятся метрики на тестовой выборке.
        """
        if use_config:
            try:
                iterations = self.config.getint("CATBOOST", "iterations")
                depth = self.config.getint("CATBOOST", "depth")
                learning_rate = self.config.getfloat("CATBOOST", "learning_rate")
            except KeyError:
                self.log.warning("Параметры для CatBoost не найдены в config.ini, используются переданные значения.")
        
        # Инициализация и обучение модели
        classifier = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            verbose=0,
            random_state=42
        )
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        
        if predict:
            try:
                y_pred = classifier.predict(self.X_test)
                acc = accuracy_score(self.y_test, y_pred)
                report = classification_report(self.y_test, y_pred)
                cm = confusion_matrix(self.y_test, y_pred)
                print("Accuracy: ", acc)
                print("Classification Report:\n", report)
                print("Confusion Matrix:\n", cm)
            except Exception:
                self.log.error(traceback.format_exc())
        
        params = {
            'iterations': iterations,
            'depth': depth,
            'learning_rate': learning_rate,
            'model_path': self.model_path
        }
        return self.save_model(classifier, self.model_path, "CATBOOST", params)

    def save_model(self, classifier, path: str, section: str, params: dict) -> bool:
        """
        Сохраняет обученную модель в файл и обновляет config.ini параметрами модели.
        """
        self.config[section] = {}
        for key, value in params.items():
            self.config[section][key] = str(value)
        
        try:
            with open('config.ini', 'w') as configfile:
                self.config.write(configfile)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(classifier, f)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        
        self.log.info(f"Модель сохранена по пути: {path}")
        return os.path.isfile(path)

if __name__ == "__main__":
    trainer = CatBoostClassifierTrainer()
    trainer.train_model(use_config=False, predict=True)
