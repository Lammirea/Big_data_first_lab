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
import uvicorn

SHOW_LOG = True

app = FastAPI()

def train_model_func(use_config: bool, n_estimators: int, max_depth: int, min_samples_split: int, predict_flag: bool):
    logger = Logger(SHOW_LOG)
    log = logger.get_logger(__name__)
    config = configparser.ConfigParser()
    config.read("../config.ini")  # Относительный путь к config.ini из api_folder
    
    try:
        split_data = config["SPLIT_DATA"]
        X_train = pd.read_csv(os.path.join("..", split_data["X_train"]), index_col=0)  # Корректировка пути к данным
        y_train = pd.read_csv(os.path.join("..", split_data["y_train"]), index_col=0)
        X_test = pd.read_csv(os.path.join("..", split_data["X_test"]), index_col=0)
        y_test = pd.read_csv(os.path.join("..", split_data["y_test"]), index_col=0)
    except Exception:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Ошибка загрузки данных из config.ini")
    
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)
    
    if use_config:
        try:
            n_estimators = config.getint("RANDOM_FOREST", "n_estimators")
            max_depth = config.getint("RANDOM_FOREST", "max_depth", fallback=None)
            min_samples_split = config.getint("RANDOM_FOREST", "min_samples_split")
        except KeyError:
            log.error(traceback.format_exc())
            log.warning("Параметры для RandomForest не найдены в config.ini. Используются переданные значения.")
    
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    try:
        classifier.fit(X_train_scaled, y_train.values.ravel())
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
            raise HTTPException(status_code=500, detail="Ошибка вычисления accuracy")
    
    project_path = os.path.join(os.getcwd(), "experiments")
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    rf_path = os.path.join(project_path, "random_forest_model.sav")
    
    params = {
        'n_estimators': n_estimators,
        'max_depth': str(max_depth),
        'min_samples_split': min_samples_split,
        'path': rf_path
    }
    
    config["RANDOM_FOREST"] = {}
    for key, value in params.items():
        config["RANDOM_FOREST"][key] = str(value)
    try:
        os.remove("../config.ini")
    except Exception:
        pass
    with open("../config.ini", "w") as configfile:
        config.write(configfile)
        
    try:
        with open(rf_path, "wb") as f:
            pickle.dump(classifier, f)
    except Exception:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Ошибка сохранения модели")
    
    log.info(f"Модель сохранена по пути: {rf_path}")
    return {"model_saved": os.path.isfile(rf_path), "test_accuracy": test_accuracy}

def predict_model_func(mode: str, file_contents: bytes = None):
    logger = Logger(SHOW_LOG)
    log = logger.get_logger(__name__)
    config = configparser.ConfigParser()
    config.read("../config.ini")
    
    try:
        model_path = config["RANDOM_FOREST"]["path"]
        with open(model_path, "rb") as f:
            classifier = pickle.load(f)
    except Exception:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Ошибка загрузки модели")
    
    if mode == "smoke":
        try:
            split_data = config["SPLIT_DATA"]
            X_test = pd.read_csv(os.path.join("..", split_data["X_test"]), index_col=0)
            y_test = pd.read_csv(os.path.join("..", split_data["y_test"]), index_col=0)
            X_train = pd.read_csv(os.path.join("..", split_data["X_train"]), index_col=0)
        except Exception:
            log.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Ошибка загрузки данных")
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
            raise HTTPException(status_code=400, detail="Файл не предоставлен")
        try:
            data = pd.read_csv(io.StringIO(file_contents.decode('utf-8')))
            split_data = config["SPLIT_DATA"]
            X_train = pd.read_csv(os.path.join("..", split_data["X_train"]), index_col=0)
        except Exception:
            log.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Ошибка обработки данных")
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
        raise HTTPException(status_code=400, detail="Неверный режим. Используйте 'smoke' или 'upload'.")

