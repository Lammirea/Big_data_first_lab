from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from logger import Logger  # Модуль logger.py должен реализовывать класс Logger
import io
import os
import configparser
import pickle
import traceback
from fastapi import HTTPException
import pandas as pd
from catboost import CatBoostClassifier

SHOW_LOG = True

def train_model_func(use_config: bool, iterations: int, depth: int, learning_rate: float, predict_flag: bool):
    """
    Обучение модели по логике train.py.
    Читаются пути к данным из config.ini (раздел SPLIT_DATA), выполняется масштабирование,
    обучается модель CatBoostClassifier, при predict_flag вычисляется accuracy на тестовой выборке.
    Затем модель сохраняется, а в config.ini обновляются параметры в секции CATBOOST.
    """
    logger = Logger(SHOW_LOG)
    log = logger.get_logger(__name__)
    config = configparser.ConfigParser()
    config.read("config.ini")
    
    # Загрузка данных из config.ini
    try:
        split_data = config["SPLIT_DATA"]
        X_train = pd.read_csv(split_data["X_train"], index_col=0)
        y_train = pd.read_csv(split_data["y_train"], index_col=0)
        X_test = pd.read_csv(split_data["X_test"], index_col=0)
        y_test = pd.read_csv(split_data["y_test"], index_col=0)
    except Exception:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Ошибка загрузки данных из config.ini")
    
    # Масштабирование
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    
    # При использовании параметров из config.ini (use_config=True)
    if use_config:
        try:
            iterations = config.getint("CATBOOST", "iterations")
            depth = config.getint("CATBOOST", "depth")
            learning_rate = config.getfloat("CATBOOST", "learning_rate")
        except KeyError:
            log.error(traceback.format_exc())
            log.warning("Параметры для CatBoost не найдены в config.ini. Используются переданные значения.")
    
    # Инициализация и обучение модели
    classifier = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        verbose=0,
        random_state=42
    )
    try:
        classifier.fit(X_train_scaled, y_train)
    except Exception:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Ошибка обучения модели")
    
    test_accuracy = None
    if predict_flag:
        try:
            y_pred = classifier.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, y_pred)
        except Exception:
            log.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Ошибка вычисления accuracy на тестовой выборке")
    
    # Определяем путь для сохранения модели
    project_path = os.path.join(os.getcwd(), "experiments")
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    catboost_path = os.path.join(project_path, "catboost_model.sav")
    
    # Параметры обученной модели
    params = {
        'iterations': iterations,
        'depth': depth,
        'learning_rate': learning_rate,
        'path': catboost_path
    }
    
    # Обновляем/добавляем секцию CATBOOST в config.ini
    config["CATBOOST"] = {}
    for key, value in params.items():
        config["CATBOOST"][key] = str(value)
    try:
        os.remove("config.ini")
    except Exception:
        pass
    with open("config.ini", "w") as configfile:
        config.write(configfile)
        
    # Сохранение модели через pickle
    try:
        with open(catboost_path, "wb") as f:
            pickle.dump(classifier, f)
    except Exception:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Ошибка сохранения модели")
    
    log.info(f"Модель сохранена по пути: {catboost_path}")
    return {"model_saved": os.path.isfile(catboost_path), "test_accuracy": test_accuracy}

def predict_model_func(mode: str, file_contents: bytes = None):
    """
    Предсказание по логике predict.py.
    Если mode=='smoke', то данные для тестирования берутся из config.ini (раздел SPLIT_DATA).
    Если mode=='upload', то используется загруженный CSV-файл.
    Для масштабирования во всех случаях применяется StandardScaler, обученный на X_train из config.ini.
    """
    logger = Logger(SHOW_LOG)
    log = logger.get_logger(__name__)
    config = configparser.ConfigParser()
    config.read("config.ini")
    
    # Загрузка сохранённой модели
    try:
        model_path = config["CATBOOST"]["path"]
        with open(model_path, "rb") as f:
            classifier = pickle.load(f)
    except Exception:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Ошибка загрузки модели из config.ini")
    
    if mode == "smoke":
        try:
            split_data = config["SPLIT_DATA"]
            X_test = pd.read_csv(split_data["X_test"], index_col=0)
            y_test = pd.read_csv(split_data["y_test"], index_col=0)
        except Exception:
            log.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Ошибка загрузки тестовых данных из config.ini")
        try:
            X_train = pd.read_csv(split_data["X_train"], index_col=0)
        except Exception:
            log.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Ошибка загрузки обучающих данных для масштабирования")
        sc = StandardScaler()
        sc.fit(X_train)
        X_test_scaled = sc.transform(X_test)
        try:
            score = classifier.score(X_test_scaled, y_test)
        except Exception:
            log.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Ошибка вычисления метрики")
        return {"mode": "smoke", "test_score": score}
    
    elif mode == "upload":
        if file_contents is None:
            raise HTTPException(status_code=400, detail="Файл не предоставлен для режима upload")
        try:
            data = pd.read_csv(io.StringIO(file_contents.decode('utf-8')))
        except Exception:
            raise HTTPException(status_code=400, detail="Ошибка чтения загруженного файла")
        try:
            split_data = config["SPLIT_DATA"]
            X_train = pd.read_csv(split_data["X_train"], index_col=0)
        except Exception:
            log.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Ошибка загрузки обучающих данных для масштабирования")
        sc = StandardScaler()
        sc.fit(X_train)
        data_scaled = sc.transform(data)
        try:
            preds = classifier.predict(data_scaled)
        except Exception:
            log.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Ошибка предсказания")
        return {"mode": "upload", "predictions": preds.tolist()}
    
    else:
        raise HTTPException(status_code=400, detail="Неверный режим предсказания. Используйте 'smoke' или 'upload'.")