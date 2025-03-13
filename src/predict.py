import argparse
import configparser
from datetime import datetime
import os
import json
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import shutil
import sys
import time
import traceback
import yaml
import numpy as np
from logger import Logger

SHOW_LOG = True

class Predictor:
    def __init__(self):
        # Инициализация логгера и конфигурации
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.ini")
        self.config.read(config_path)
        
        # Парсер аргументов командной строки
        self.parser = argparse.ArgumentParser(description="Predictor")
        self.parser.add_argument("-m", "--model", type=str, help="Select model", required=True, default="D_TREE",
                                 const="D_TREE", nargs="?", choices=["LOG_REG", "RAND_FOREST", "KNN", "GNB", "SVM", "D_TREE"])
        self.parser.add_argument("-t", "--tests", type=str, help="Select tests", required=True, default="smoke",
                                 const="smoke", nargs="?", choices=["smoke", "func"])
        
        # Загрузка данных из файлов, указанных в config.ini
        project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
        train_path = os.path.normpath(os.path.join(project_root, self.config["DATA"]["train_file"]))
        test_path = os.path.normpath(os.path.join(project_root, self.config["DATA"]["test_file"]))
        train_df = pd.read_csv(train_path, encoding='latin1', low_memory=False)
        test_df = pd.read_csv(test_path, encoding='latin1', low_memory=False)
        
        # Предобработка данных
        X_train_raw, self.y_train = self.preprocess_data(train_df)
        X_test_raw, self.y_test = self.preprocess_data(test_df)
        self.feature_columns = list(X_train_raw.columns)
        
        # Создание pipeline для предобработки
        self.pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        self.X_train_scaled = self.pipeline.fit_transform(X_train_raw)
        self.X_test_scaled = self.pipeline.transform(X_test_raw)
        
        # Путь для сохранения предобработчика
        self.project_path = os.path.join(os.getcwd(), "experiments")
        if not os.path.exists(self.project_path):
            os.makedirs(self.project_path)
        self.preprocessor_path = os.path.join(self.project_path, "preprocessor.sav")
        with open(self.preprocessor_path, "wb") as f:
            pickle.dump({'pipeline': self.pipeline, 'feature_columns': self.feature_columns}, f)
        
        self.log.info("Predictor is ready")

    def preprocess_data(self, df):
        # Удаление ненужных столбцов
        columns_to_drop_cat = [' Source IP', ' Destination IP', ' Timestamp', 'Flow ID', ' Label']
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
        # Создание целевой переменной
        df['State'] = df[' Label'].map(lambda a: 1 if a in ['BENIGN'] else 0)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = df.drop(columns=columns_to_drop_cat + columns_to_drop + ['State'], errors='ignore')
        y = df['State']
        return X, y

    def predict(self):
        args = self.parser.parse_args()
        try:
            model_path = self.config[args.model]["path"]
            with open(model_path, "rb") as f:
                classifier = pickle.load(f)
        except (FileNotFoundError, KeyError):
            self.log.error(traceback.format_exc())
            sys.exit(1)
        
        if args.tests == "smoke":
            try:
                score = classifier.score(self.X_test_scaled, self.y_test)
                print(f'{args.model} has {score} score')
            except Exception:
                self.log.error(traceback.format_exc())
                sys.exit(1)
            self.log.info(f'{model_path} passed smoke tests')
        
        elif args.tests == "func":
            tests_path = os.path.join(os.getcwd(), "tests")
            exp_path = os.path.join(os.getcwd(), "experiments")
            for test in os.listdir(tests_path):
                with open(os.path.join(tests_path, test)) as f:
                    try:
                        data = json.load(f)
                        X_raw = pd.json_normalize(data, record_path=['X'])
                        y = pd.json_normalize(data, record_path=['y'])
                        X_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
                        X_scaled = self.pipeline.transform(X_raw[self.feature_columns])
                        score = classifier.score(X_scaled, y)
                        print(f'{args.model} has {score} score')
                    except Exception:
                        self.log.error(traceback.format_exc())
                        sys.exit(1)
                    self.log.info(f'{model_path} passed func test {test}')
                    
                    # Сохранение результатов эксперимента
                    exp_data = {
                        "model": args.model,
                        "model_params": dict(self.config.items(args.model)),
                        "tests": args.tests,
                        "score": str(score),
                        "X_test_path": test,
                        "y_test_path": test,
                    }
                    date_time = datetime.fromtimestamp(time.time())
                    str_date_time = date_time.strftime("%Y_%m_%d_%H_%M_%S")
                    exp_dir = os.path.join(exp_path, f'exp_{test[:6]}_{str_date_time}')
                    os.mkdir(exp_dir)
                    with open(os.path.join(exp_dir, "exp_config.yaml"), 'w') as exp_f:
                        yaml.safe_dump(exp_data, exp_f, sort_keys=False)
                    shutil.copy(os.path.join(os.getcwd(), "logfile.log"), os.path.join(exp_dir, "exp_logfile.log"))
                    shutil.copy(model_path, os.path.join(exp_dir, f'exp_{args.model}.sav'))
        return True

if __name__ == "__main__":
    predictor = Predictor()
    predictor.predict()