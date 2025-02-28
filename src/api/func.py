from fastapi import FastAPI, HTTPException, UploadFile, File
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from logger import Logger  # Предполагается наличие модуля logger.py
import io
import os
import configparser
import pickle
import traceback
import pandas as pd

SHOW_LOG = True

app = FastAPI()

def train_model_func(use_config: bool, n_estimators: int, max_depth: int, min_samples_split: int, predict_flag: bool):
    """
    Обучение модели RandomForestClassifier.
    Читаются пути к данным из config.ini (раздел SPLIT_DATA), выполняется масштабирование,
    обучается модель, при predict_flag вычисляется accuracy на тестовой выборке.
    Затем модель сохраняется, а в config.ini обновляются параметры в секции RANDOM_FOREST.
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
    
    # Масштабирование (опционально для RandomForest, но оставлено для совместимости)
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    
    # При использовании параметров из config.ini (use_config=True)
    if use_config:
        try:
            n_estimators = config.getint("RANDOM_FOREST", "n_estimators")
            max_depth = config.getint("RANDOM_FOREST", "max_depth", fallback=None)
            min_samples_split = config.getint("RANDOM_FOREST", "min_samples_split")
        except KeyError:
            log.error(traceback.format_exc())
            log.warning("Параметры для RandomForest не найдены в config.ini. Используются переданные значения.")
    
    # Инициализация и обучение модели
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    try:
        classifier.fit(X_train_scaled, y_train.values.ravel())  # .values.ravel() для преобразования y_train
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
    rf_path = os.path.join(project_path, "random_forest_model.sav")
    
    # Параметры обученной модели
    params = {
        'n_estimators': n_estimators,
        'max_depth': str(max_depth),  # Преобразуем в строку для корректной записи None
        'min_samples_split': min_samples_split,
        'path': rf_path
    }
    
    # Обновляем/добавляем секцию RANDOM_FOREST в config.ini
    config["RANDOM_FOREST"] = {}
    for key, value in params.items():
        config["RANDOM_FOREST"][key] = str(value)
    try:
        os.remove("config.ini")
    except Exception:
        pass
    with open("config.ini", "w") as configfile:
        config.write(configfile)
        
    # Сохранение модели через pickle
    try:
        with open(rf_path, "wb") as f:
            pickle.dump(classifier, f)
    except Exception:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Ошибка сохранения модели")
    
    log.info(f"Модель сохранена по пути: {rf_path}")
    return {"model_saved": os.path.isfile(rf_path), "test_accuracy": test_accuracy}

def predict_model_func(mode: str, file_contents: bytes = None):
    """
    Предсказание с использованием RandomForestClassifier.
    Если mode=='smoke', данные для тестирования берутся из config.ini (раздел SPLIT_DATA).
    Если mode=='upload', используется загруженный CSV-файл.
    """
    logger = Logger(SHOW_LOG)
    log = logger.get_logger(__name__)
    config = configparser.ConfigParser()
    config.read("config.ini")
    
    # Загрузка сохранённой модели
    try:
        model_path = config["RANDOM_FOREST"]["path"]
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

