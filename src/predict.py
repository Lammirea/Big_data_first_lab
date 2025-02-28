import argparse
import configparser
from datetime import datetime
import os
import json
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import shutil
import sys
import time
import traceback
import yaml

from logger import Logger

SHOW_LOG = True


class Predictor:
    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")
        self.parser = argparse.ArgumentParser(
            description="RandomForestClassifier Predictor for Anomaly Traffic"
        )
        self.parser.add_argument(
            "-m",
            "--model",
            type=str,
            help="Select model",
            required=True,
            default="RANDOM_FOREST",
            const="RANDOM_FOREST",
            nargs="?",
            choices=["RANDOM_FOREST"]
        )
        self.parser.add_argument(
            "-t",
            "--tests",
            type=str,
            help="Select test mode",
            required=True,
            default="smoke",
            const="smoke",
            nargs="?",
            choices=["smoke", "func"]
        )
        try:
            split_data = self.config["SPLIT_DATA"]
            self.X_train = pd.read_csv(split_data["X_train"], index_col=0)
            self.y_train = pd.read_csv(split_data["y_train"], index_col=0)
            self.X_test = pd.read_csv(split_data["X_test"], index_col=0)
            self.y_test = pd.read_csv(split_data["y_test"], index_col=0)
        except Exception as e:
            self.log.error("Ошибка загрузки данных из config.ini: " + str(e))
            sys.exit(1)

        # Для RandomForest масштабирование не обязательно, но оставляем для единообразия
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(self.X_train)
        self.X_test = self.sc.transform(self.X_test)
        self.log.info("Predictor is ready")

    def predict(self) -> bool:
        args = self.parser.parse_args()
        try:
            model_section = args.model  # Ожидается "RANDOM_FOREST"
            model_path = self.config[model_section]["path"]
            classifier = pickle.load(open(model_path, "rb"))
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)

        if args.tests == "smoke":
            try:
                score = classifier.score(self.X_test, self.y_test)
                print(f'{args.model} has {score} score on test set')
            except Exception:
                self.log.error(traceback.format_exc())
                sys.exit(1)
            self.log.info(f'{model_path} passed smoke tests')

        elif args.tests == "func":
            tests_path = os.path.join(os.getcwd(), "tests")
            exp_path = os.path.join(os.getcwd(), "experiments")
            for test in os.listdir(tests_path):
                test_file = os.path.join(tests_path, test)
                with open(test_file) as f:
                    try:
                        data = json.load(f)
                        # Предполагается, что JSON содержит списки объектов под ключами 'X' и 'y'
                        X = self.sc.transform(pd.json_normalize(data, record_path=['X']))
                        y = pd.json_normalize(data, record_path=['y'])
                        score = classifier.score(X, y)
                        print(f'{args.model} has {score} score on functional test {test}')
                    except Exception:
                        self.log.error(traceback.format_exc())
                        sys.exit(1)
                    self.log.info(f'{model_path} passed functional test {test_file}')
                    exp_data = {
                        "model": args.model,
                        "model params": dict(self.config.items(args.model)),
                        "tests": args.tests,
                        "score": str(score),
                        "X_test path": self.config["SPLIT_DATA"]["X_test"],
                        "y_test path": self.config["SPLIT_DATA"]["y_test"],
                    }
                    date_time = datetime.fromtimestamp(time.time())
                    str_date_time = date_time.strftime("%Y_%m_%d_%H_%M_%S")
                    exp_dir = os.path.join(exp_path, f'exp_{test[:6]}_{str_date_time}')
                    os.mkdir(exp_dir)
                    with open(os.path.join(exp_dir, "exp_config.yaml"), 'w') as exp_f:
                        yaml.safe_dump(exp_data, exp_f, sort_keys=False)
                    shutil.copy(os.path.join(os.getcwd(), "logfile.log"),
                                os.path.join(exp_dir, "exp_logfile.log"))
                    shutil.copy(model_path, os.path.join(exp_dir, f'exp_{args.model}.sav'))
        return True


if __name__ == "__main__":
    predictor = Predictor()
    predictor.predict()