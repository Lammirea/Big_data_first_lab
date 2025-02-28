import configparser
import os
import pandas as pd
import pickle
import sys
import traceback
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from logger import Logger  # Предполагается, что модуль logger.py реализует класс Logger

SHOW_LOG = True

class RandomForestModel:
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
        
        # Применяем масштабирование (для RandomForest не обязательно, но оставлено для совместимости)
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)
        
        # Определяем путь для сохранения модели
        self.project_path = os.path.join(os.getcwd(), "experiments")
        if not os.path.exists(self.project_path):
            os.makedirs(self.project_path)
        self.rf_path = os.path.join(self.project_path, "random_forest_model.sav")
        
        self.log.info("RandomForestModel is ready")

    def rf_train(self, use_config: bool = False, n_estimators: int = 100, max_depth: int = None, 
                min_samples_split: int = 2, predict: bool = False) -> bool:
        """
        Обучает модель RandomForestClassifier.
        
        Если use_config=True, пытается получить параметры из секции [RANDOM_FOREST] файла config.ini,
        иначе использует переданные параметры.
        
        Если predict=True, выводит accuracy на тестовой выборке.
        """
        if use_config:
            try:
                n_estimators = self.config.getint("RANDOM_FOREST", "n_estimators")
                max_depth = self.config.getint("RANDOM_FOREST", "max_depth", fallback=None)
                min_samples_split = self.config.getint("RANDOM_FOREST", "min_samples_split")
            except KeyError:
                self.log.error(traceback.format_exc())
                self.log.warning("Параметры для RandomForest не найдены в config.ini. Используются переданные значения.")
        
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
            y_pred = classifier.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            print(f"Test Accuracy: {acc}")
        
        params = {
            'n_estimators': n_estimators,
            'max_depth': str(max_depth),  # Преобразуем в строку, так как может быть None
            'min_samples_split': min_samples_split,
            'path': self.rf_path
        }
        return self.save_model(classifier, self.rf_path, "RANDOM_FOREST", params)

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
    model = RandomForestModel()
    # Обучение модели с параметрами по умолчанию (или с параметрами из config.ini, если use_config=True)
    model.rf_train(use_config=False, predict=True)