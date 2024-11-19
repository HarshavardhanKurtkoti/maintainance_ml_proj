import unittest
from data.preprocessing import preprocess_data
import pandas as pd
from data.data_loader import load_data
from modeling.model import train_model
from config.settings import config

class TestDataLoader(unittest.TestCase):
    def test_load_data(self):
        data = load_data(config.DATA_PATH)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)

class TestPreprocessing(unittest.TestCase):
    def test_preprocess_data(self):
        data = load_data(config.DATA_PATH)
        preprocessed_data = preprocess_data(data)
        self.assertEqual(preprocessed_data.isnull().sum().sum(), 0)
        self.assertGreater(preprocessed_data.shape[0], 0)

class TestModelTraining(unittest.TestCase):
    def test_train_model(self):
        data = load_data(config.DATA_PATH)
        preprocessed_data = preprocess_data(data)
        model = train_model(preprocessed_data)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))

if __name__ == "__main__":
    unittest.main()
