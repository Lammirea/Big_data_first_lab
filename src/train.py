import configparser
import os
import pandas as pd
import pickle
import sys
import traceback
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from logger import Logger  # Предполагается, что модуль logger.py реализует класс Logger

SHOW_LOG = True

class CatBoostModel:
    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")
        
        # Загрузка данных (пути к файлам должны быть указаны в секции SPLIT_DATA)
        try:
            self.X_train = pd.read_csv(self.config["SPLIT_DATA"]["X_train"], index_col=0)
            self.y_train = pd.read_csv(self.config["SPLIT_DATA"]["y_train"], index_col=0)
            self.X_test = pd.read_csv(self.config["SPLIT_DATA"]["X_test"], index_col=0)
            self.y_test = pd.read_csv(self.config["SPLIT_DATA"]["y_test"], index_col=0)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        
        # Применяем масштабирование (опционально – для CatBoost оно не обязательно)
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)
        
        # Определяем путь для сохранения модели
        self.project_path = os.path.join(os.getcwd(), "experiments")
        if not os.path.exists(self.project_path):
            os.makedirs(self.project_path)
        self.catboost_path = os.path.join(self.project_path, "catboost_model.sav")
        
        self.log.info("CatBoostModel is ready")

    def catboost_train(self, use_config: bool = False, iterations: int = 1000, depth: int = 6, learning_rate: float = 0.1, predict: bool = False) -> bool:
        """
        Обучает модель CatBoostClassifier.
        
        Если use_config=True, пытается получить параметры из секции [CATBOOST] файла config.ini,
        иначе использует переданные параметры.
        
        Если predict=True, выводит accuracy на тестовой выборке.
        """
        if use_config:
            try:
                iterations = self.config.getint("CATBOOST", "iterations")
                depth = self.config.getint("CATBOOST", "depth")
                learning_rate = self.config.getfloat("CATBOOST", "learning_rate")
            except KeyError:
                self.log.error(traceback.format_exc())
                self.log.warning("Параметры для CatBoost не найдены в config.ini. Используются переданные значения.")
        
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
            y_pred = classifier.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            print(f"Test Accuracy: {acc}")
        
        params = {
            'iterations': iterations,
            'depth': depth,
            'learning_rate': learning_rate,
            'path': self.catboost_path
        }
        return self.save_model(classifier, self.catboost_path, "CATBOOST", params)

    def save_model(self, classifier, path: str, name: str, params: dict) -> bool:
        """
        Сохраняет обученную модель в файл и обновляет config.ini параметрами модели.
        """
        # Добавляем/обновляем секцию с параметрами обученной модели
        self.config[name] = {}
        for key, value in params.items():
            self.config[name][key] = str(value)
        
        # Перезаписываем config.ini
        try:
            os.remove("config.ini")
        except Exception:
            pass
        with open("config.ini", "w") as configfile:
            self.config.write(configfile)
        
        # Сохраняем модель с помощью pickle
        try:
            with open(path, "wb") as f:
                pickle.dump(classifier, f)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        
        self.log.info(f"Модель сохранена по пути: {path}")
        return os.path.isfile(path)


if __name__ == "__main__":
    model = CatBoostModel()
    # Обучение модели с параметрами по умолчанию (или с параметрами из config.ini, если use_config=True)
    model.catboost_train(use_config=False, predict=True)
