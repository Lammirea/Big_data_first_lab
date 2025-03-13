import configparser
import os
import pandas as pd
import numpy as np
import sys
import traceback

from logger import Logger

SHOW_LOG = True

class DataMaker:
    def __init__(self):
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.ini")
        self.config.read(config_path)
        print("Секции конфига:", self.config.sections())

        self.project_path = os.path.join(os.getcwd(), "data")
        # Пути для сохранения предобработанных данных
        self.train_path = [
            os.path.join(self.project_path, "preprocessed_train_X.csv"),
            os.path.join(self.project_path, "preprocessed_train_y.csv")
        ]
        self.test_path = [
            os.path.join(self.project_path, "preprocessed_test_X.csv"),
            os.path.join(self.project_path, "preprocessed_test_y.csv")
        ]
        self.log.info("DataMaker is ready")

    def preprocess_data(self, df):
        """Предобработка данных о сетевом трафике."""
        # Список столбцов для удаления
        columns_to_drop_cat = [' Source IP', ' Destination IP', ' Timestamp', 'Flow ID']
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
        # Создание целевой переменной: 1 - BENIGN, 0 - атака
        df['State'] = df[' Label'].map(lambda a: 1 if a == 'BENIGN' else 0)
        # Замена бесконечных значений на NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # Удаление ненужных столбцов и столбцов ' Label', 'State' из X
        X = df.drop(columns=columns_to_drop_cat + columns_to_drop + [' Label', 'State'], errors='ignore')
        y = df['State']
        return X, y

    def prepare_data(self):
        """Загрузка, предобработка и сохранение данных."""
        # Загрузка обучающих данных
        train_path = os.path.normpath(os.path.join(self.project_path, self.config["DATA"]["train_file"]))
        train_df = pd.read_csv(train_path, encoding='latin1', low_memory=False)
        X_train, y_train = self.preprocess_data(train_df)
        # Сохранение предобработанных обучающих данных
        X_train.to_csv(self.train_path[0], index=True)
        y_train.to_csv(self.train_path[1], index=True)

        # Загрузка тестовых данных
        test_path = os.path.normpath(os.path.join(self.project_path, self.config["DATA"]["test_file"]))
        test_df = pd.read_csv(test_path, encoding='latin1', low_memory=False)
        X_test, y_test = self.preprocess_data(test_df)
        # Сохранение предобработанных тестовых данных
        X_test.to_csv(self.test_path[0], index=True)
        y_test.to_csv(self.test_path[1], index=True)

        # Обновление конфигурации
        self.config["PREPROCESSED_DATA"] = {
            'X_train': self.train_path[0],
            'y_train': self.train_path[1],
            'X_test': self.test_path[0],
            'y_test': self.test_path[1]
        }
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        self.log.info("Предобработанные обучающие и тестовые данные сохранены")
        return True

if __name__ == "__main__":
    data_maker = DataMaker()
    data_maker.prepare_data()