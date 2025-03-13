from fastapi import FastAPI, HTTPException, UploadFile, File
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from logger import Logger  # Предполагается наличие модуля logger.py
import io
import os
import configparser
import pickle
import traceback
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np

SHOW_LOG = True

app = FastAPI()

current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))

# Список столбцов для удаления (из оригинального кода)
columns_to_drop = [
    'Total Fwd Packets', 'Flow IAT Mean', 'Fwd Packet Length Std', 'Bwd IAT Mean',
    'Bwd IAT Max', 'Fwd IAT Total', 'Bwd IAT Mean', 'Active Max', 'Fwd IAT Min',
    'Fwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Total', 'Fwd PSH Flags', 'FIN Flag Count',
    'Active Min', 'Down/Up Ratio', 'Bwd IAT Min', 'Active Std', 'Fwd Packet Length Min',
    'SYN Flag Count', 'Active Mean', 'Idle Std', 'Bwd PSH Flags', 'Bwd URG Flags',
    'Fwd URG Flags', 'Fwd Avg Bytes/Bulk', 'RST Flag Count', 'CWE Flag Count',
    'Bwd Avg Bulk Rate', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bytes/Bulk',
    'Fwd Avg Bulk Rate', 'Fwd Avg Packets/Bulk', 'ECE Flag Count'
]

def preprocess_data(df):
    """Предобработка данных: отображение меток, удаление столбцов, выделение X и y."""
    # Отображение ' Label' в 'State' (BENIGN = 1, атаки = 0)
    df['State'] = df[' Label'].map(lambda a: 1 if a == 'BENIGN' else 0)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Удаление ненужных столбцов и 'State'
    X = df.drop(columns=columns_to_drop + ['State'], errors='ignore')
    y = df['State']
    return X, y

def train_model_func(use_config: bool, max_depth: int, min_samples_split: int, predict_flag: bool):
    logger = Logger(SHOW_LOG)
    log = logger.get_logger(__name__)
    config = configparser.ConfigParser()
    config_path = os.path.join(current_dir, '..', "config.ini")
    config.read(config_path, encoding="utf-8")

    # Загрузка тренировочных данных
    try:
        train_path = os.path.normpath(os.path.join(project_root, config["DATA"]["train_file"]))
        train_df = pd.read_csv(train_path, encoding='latin1', low_memory=False)
    except Exception:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Ошибка загрузки тренировочных данных")

    # Предобработка тренировочных данных
    X_train_raw, y_train = preprocess_data(train_df)
    feature_columns = list(X_train_raw.columns)  # Сохраняем список признаков

    # Создание предобработчика
    categorical_columns = X_train_raw.select_dtypes(include=['object']).columns
    preprocessor = ColumnTransformer(
        transformers=[('cat', OrdinalEncoder(), categorical_columns)],
        remainder='passthrough'
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Применение предобработки
    try:
        X_train_scaled = pipeline.fit_transform(X_train_raw)
    except Exception:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Ошибка предобработки данных")

    # Применение SMOTE
    smote = SMOTE(random_state=42)
    try:
        X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    except Exception:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Ошибка применения SMOTE")

    # Настройка параметров модели
    if use_config:
        try:
            max_depth = config.getint("DECISION_TREE", "max_depth", fallback=max_depth)
            min_samples_split = config.getint("DECISION_TREE", "min_samples_split", fallback=min_samples_split)
        except Exception:
            log.warning("Параметры для DecisionTree не найдены в config.ini. Используются переданные значения.")

    # Обучение модели
    classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    try:
        classifier.fit(X_train_smote, y_train_smote)
    except Exception:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Ошибка обучения модели")

    # Сохранение модели и предобработчика
    project_path = os.path.join(os.getcwd(), "experiments")
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    model_path = os.path.join(project_path, "decision_tree_model.sav")
    preprocessor_path = os.path.join(project_path, "preprocessor.sav")

    try:
        with open(model_path, "wb") as f:
            pickle.dump(classifier, f)
        with open(preprocessor_path, "wb") as f:
            pickle.dump({'pipeline': pipeline, 'feature_columns': feature_columns}, f)
    except Exception:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Ошибка сохранения модели или предобработчика")

    # Обновление config.ini
    config["DECISION_TREE"] = {
        'max_depth': str(max_depth),
        'min_samples_split': str(min_samples_split),
        'path': model_path
    }
    try:
        os.remove(config_path)
    except Exception:
        pass
    with open(config_path, "w", encoding="utf-8") as configfile:
        config.write(configfile)

    # Оценка на тестовых данных, если требуется
    test_accuracy = None
    if predict_flag:
        try:
            test_path = os.path.normpath(os.path.join(project_root, config["DATA"]["test_file"]))
            test_df = pd.read_csv(test_path, encoding='latin1', sep=",")
            X_test_raw, y_test = preprocess_data(test_df)
            X_test_scaled = pipeline.transform(X_test_raw[feature_columns])
            y_pred = classifier.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, y_pred)
        except Exception:
            log.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Ошибка оценки модели")

    log.info(f"Модель сохранена по пути: {model_path}")
    return {"model_saved": os.path.isfile(model_path), "test_accuracy": test_accuracy}

def predict_model_func(mode: str, file_contents: bytes = None):
    logger = Logger(SHOW_LOG)
    log = logger.get_logger(__name__)
    config = configparser.ConfigParser()
    config_path = os.path.join(current_dir, '..', "config.ini")
    config.read(config_path, encoding="utf-8")

    # Загрузка модели и предобработчика
    try:
        model_path = config["DECISION_TREE"]["path"]
        with open(model_path, "rb") as f:
            classifier = pickle.load(f)
        preprocessor_path = os.path.join(project_root, "experiments", "preprocessor.sav")
        with open(preprocessor_path, "rb") as f:
            preproc_data = pickle.load(f)
            pipeline = preproc_data['pipeline']
            feature_columns = preproc_data['feature_columns']
    except Exception:
        log.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Ошибка загрузки модели или предобработчика")

    if mode == "smoke":
        # Тестовый прогон на тестовых данных
        try:
            test_path = os.path.normpath(os.path.join(project_root, config["DATA"]["test_file"]))
            test_df = pd.read_csv(test_path, encoding='latin1', sep=",")
            X_test_raw, y_test = preprocess_data(test_df)
            X_test_scaled = pipeline.transform(X_test_raw[feature_columns])
            score = classifier.score(X_test_scaled, y_test)
        except Exception:
            log.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Ошибка тестового прогона")
        return {"mode": "smoke", "test_score": score}

    elif mode == "upload":
        if file_contents is None:
            raise HTTPException(status_code=400, detail="Файл не предоставлен")
        try:
            uploaded_df = pd.read_csv(io.StringIO(file_contents.decode('utf-8')))
            X_upload_raw = uploaded_df[feature_columns]  # Предполагается, что файл содержит те же признаки
            X_upload_scaled = pipeline.transform(X_upload_raw)
            preds = classifier.predict(X_upload_scaled)
        except Exception:
            log.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Ошибка обработки загруженного файла")
        return {"mode": "upload", "predictions": preds.tolist()}

    else:
        raise HTTPException(status_code=400, detail="Неверный режим. Используйте 'smoke' или 'upload'.")
