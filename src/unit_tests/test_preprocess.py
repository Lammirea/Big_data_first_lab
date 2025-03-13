import configparser
import os
import unittest
import pandas as pd
import sys

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

current_dir = os.path.dirname(__file__)
config = configparser.ConfigParser()
config_path = os.path.join(current_dir, '..', "config.ini")
config.read(config_path, encoding="utf-8")


from train import MultiModel

class TestMultiModel(unittest.TestCase):
    def setUp(self):
        """Инициализация перед каждым тестом."""
        self.model = MultiModel()
        # Тестовый DataFrame с данными
        self.test_df = pd.DataFrame({
            ' Source IP': ['192.168.1.1'],
            ' Destination IP': ['10.0.0.1'],
            ' Timestamp': ['2023-01-01 00:00:00'],
            'Flow ID': ['flow1'],
            ' Label': ['BENIGN'],
            'Total Fwd Packets': [1],
            'Flow IAT Mean': [0.1],
            # Добавьте другие столбцы, если они используются в preprocess_data
        })

    def test_preprocess_data(self):
        """Тест функции предобработки данных."""
        X, y = self.model.preprocess_data(self.test_df)
        # Проверяем, что целевая переменная правильно создана (1 для BENIGN)
        self.assertEqual(y.iloc[0], 1)
        # Проверяем, что ненужные столбцы удалены
        self.assertNotIn(' Source IP', X.columns)
        self.assertNotIn('Total Fwd Packets', X.columns)

    def test_predict_smoke(self):
        """Тест предсказания в режиме 'smoke'."""
        # Сначала обучаем модель
        self.model.d_tree(use_config=False, predict=False)
        # Выполняем предсказание
        result = self.model.predict("d_tree", "smoke")
        # Проверяем, что результат содержит test_score
        self.assertIn("test_score", result)
        # Проверяем, что test_score в допустимом диапазоне
        self.assertGreaterEqual(result["test_score"], 0.0)
        self.assertLessEqual(result["test_score"], 1.0)

if __name__ == '__main__':
    unittest.main()