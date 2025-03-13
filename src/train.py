import configparser
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import sys
import traceback
import numpy as np
import io
from logger import Logger

SHOW_LOG = True

class MultiModel:
    def __init__(self):
        # Инициализация логгера и конфигурации
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.ini")
        self.config.read(config_path)
        
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
        X_train_scaled = self.pipeline.fit_transform(X_train_raw)
        self.X_test_scaled = self.pipeline.transform(X_test_raw)
        
        # Балансировка классов с помощью SMOTE
        smote = SMOTE(random_state=42)
        self.X_train_smote, self.y_train_smote = smote.fit_resample(X_train_scaled, self.y_train)
        
        # Путь для сохранения моделей и предобработчика
        self.project_path = os.path.join(os.getcwd(), "experiments")
        if not os.path.exists(self.project_path):
            os.makedirs(self.project_path)
        self.preprocessor_path = os.path.join(self.project_path, "preprocessor.sav")
        with open(self.preprocessor_path, "wb") as f:
            pickle.dump({'pipeline': self.pipeline, 'feature_columns': self.feature_columns}, f)
        
        # Пути для сохранения моделей
        self.log_reg_path = os.path.join(self.project_path, "log_reg.sav")
        self.rand_forest_path = os.path.join(self.project_path, "rand_forest.sav")
        self.knn_path = os.path.join(self.project_path, "knn.sav")
        self.svm_path = os.path.join(self.project_path, "svm.sav")
        self.gnb_path = os.path.join(self.project_path, "gnb.sav")
        self.d_tree_path = os.path.join(self.project_path, "d_tree.sav")
        
        self.log.info("MultiModel is ready")

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

    def log_reg(self, predict=False):
        classifier = LogisticRegression()
        classifier.fit(self.X_train_smote, self.y_train_smote)
        if predict:
            y_pred = classifier.predict(self.X_test_scaled)
            print(accuracy_score(self.y_test, y_pred))
        params = {'path': self.log_reg_path}
        return self.save_model(classifier, self.log_reg_path, "LOG_REG", params)

    def rand_forest(self, use_config: bool, n_estimators=100, criterion="entropy", predict=False):
        if use_config:
            n_estimators = self.config.getint("RAND_FOREST", "n_estimators", fallback=n_estimators)
            criterion = self.config["RAND_FOREST"].get("criterion", criterion)
        classifier = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
        classifier.fit(self.X_train_smote, self.y_train_smote)
        if predict:
            y_pred = classifier.predict(self.X_test_scaled)
            print(accuracy_score(self.y_test, y_pred))
        params = {'n_estimators': str(n_estimators), 'criterion': criterion, 'path': self.rand_forest_path}
        return self.save_model(classifier, self.rand_forest_path, "RAND_FOREST", params)

    def knn(self, use_config: bool, n_neighbors=5, metric="minkowski", p=2, predict=False):
        if use_config:
            n_neighbors = self.config.getint("KNN", "n_neighbors", fallback=n_neighbors)
            metric = self.config["KNN"].get("metric", metric)
            p = self.config.getint("KNN", "p", fallback=p)
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, p=p)
        classifier.fit(self.X_train_smote, self.y_train_smote)
        if predict:
            y_pred = classifier.predict(self.X_test_scaled)
            print(accuracy_score(self.y_test, y_pred))
        params = {'n_neighbors': str(n_neighbors), 'metric': metric, 'p': str(p), 'path': self.knn_path}
        return self.save_model(classifier, self.knn_path, "KNN", params)

    def svm(self, use_config: bool, kernel="linear", random_state=0, predict=False):
        if use_config:
            kernel = self.config["SVM"].get("kernel", kernel)
            random_state = self.config.getint("SVM", "random_state", fallback=random_state)
        classifier = SVC(kernel=kernel, random_state=random_state)
        classifier.fit(self.X_train_smote, self.y_train_smote)
        if predict:
            y_pred = classifier.predict(self.X_test_scaled)
            print(accuracy_score(self.y_test, y_pred))
        params = {'kernel': kernel, 'random_state': str(random_state), 'path': self.svm_path}
        return self.save_model(classifier, self.svm_path, "SVM", params)

    def gnb(self, predict=False):
        classifier = GaussianNB()
        classifier.fit(self.X_train_smote, self.y_train_smote)
        if predict:
            y_pred = classifier.predict(self.X_test_scaled)
            print(accuracy_score(self.y_test, y_pred))
        params = {'path': self.gnb_path}
        return self.save_model(classifier, self.gnb_path, "GNB", params)

    def d_tree(self, use_config: bool, max_depth=10, min_samples_split=2, predict=False):
        if use_config:
            max_depth = self.config.getint("DECISION_TREE", "max_depth", fallback=max_depth)
            min_samples_split = self.config.getint("DECISION_TREE", "min_samples_split", fallback=min_samples_split)
        classifier = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        classifier.fit(self.X_train_smote, self.y_train_smote)
        if predict:
            y_pred = classifier.predict(self.X_test_scaled)
            print(accuracy_score(self.y_test, y_pred))
        params = {'max_depth': str(max_depth), 'min_samples_split': str(min_samples_split), 'path': self.d_tree_path}
        return self.save_model(classifier, self.d_tree_path, "DECISION_TREE", params)

    def save_model(self, classifier, path, section, params):
        # Сохранение модели и обновление конфигурации
        self.config[section] = params
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        with open(path, 'wb') as f:
            pickle.dump(classifier, f)
        self.log.info(f'{path} is saved')
        return os.path.isfile(path)

    def predict(self, model_name, mode, file_contents=None):
        # Загрузка предобработчика
        with open(self.preprocessor_path, "rb") as f:
            preproc_data = pickle.load(f)
            pipeline = preproc_data['pipeline']
            feature_columns = preproc_data['feature_columns']
        
        # Загрузка модели
        model_path = getattr(self, f"{model_name}_path")
        with open(model_path, "rb") as f:
            classifier = pickle.load(f)
        
        if mode == "smoke":
            # Предсказание на тестовых данных
            y_pred = classifier.predict(self.X_test_scaled)
            score = accuracy_score(self.y_test, y_pred)
            return {"mode": "smoke", "test_score": score}
        elif mode == "upload":
            # Предсказание на загруженном файле
            if file_contents is None:
                raise ValueError("Файл не предоставлен")
            uploaded_df = pd.read_csv(io.StringIO(file_contents.decode('utf-8')))
            uploaded_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_upload_raw = uploaded_df[feature_columns]
            X_upload_scaled = pipeline.transform(X_upload_raw)
            preds = classifier.predict(X_upload_scaled)
            return {"mode": "upload", "predictions": preds.tolist()}
        else:
            raise ValueError("Неверный режим. Используйте 'smoke' или 'upload'.")

if __name__ == "__main__":
    multi_model = MultiModel()
    multi_model.d_tree(use_config=False, predict=True)
    # Пример предсказания
    result = multi_model.predict("d_tree", "smoke")
    print(result)