import configparser
import os
import pandas as pd
import sys
import traceback
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from logger import Logger

SHOW_LOG = True

class RandomForestClassifierTrainer:
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
        self.model_path = os.path.join(self.project_path, "random_forest_model.sav")
        self.log.info("RandomForestClassifierTrainer is ready")

    def train_model(self, use_config: bool = False, n_estimators: int = 100, max_depth: int = None, 
                   min_samples_split: int = 2, predict: bool = False) -> bool:
        """
        Обучает модель RandomForestClassifier.
        
        Если use_config=True, параметры берутся из секции [RANDOM_FOREST] файла config.ini.
        Если predict=True, выводятся метрики на тестовой выборке.
        """
        if use_config:
            try:
                n_estimators = self.config.getint("RANDOM_FOREST", "n_estimators")
                max_depth = self.config.getint("RANDOM_FOREST", "max_depth", fallback=None)
                min_samples_split = self.config.getint("RANDOM_FOREST", "min_samples_split")
            except KeyError:
                self.log.warning("Параметры для RandomForest не найдены в config.ini, используются переданные значения.")
        
        # Инициализация и обучение модели
        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        try:
            classifier.fit(self.X_train, self.y_train.values.ravel())  # .values.ravel() для преобразования y_train
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
            'n_estimators': n_estimators,
            'max_depth': str(max_depth),  # Преобразуем в строку для корректной записи None
            'min_samples_split': min_samples_split,
            'model_path': self.model_path
        }
        return self.save_model(classifier, self.model_path, "RANDOM_FOREST", params)

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
    trainer = RandomForestClassifierTrainer()
    trainer.train_model(use_config=False, predict=True)