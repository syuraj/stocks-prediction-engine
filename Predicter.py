from environs import Env
from stocker import Stocker
import pandas as pd

env = Env()
api_key = env('alphavantage_api_key')

class Predicter():

    @staticmethod
    def predict():
        stock = Stocker(api_key,  'TSLA')

        print('Running prediction for TSLA')

        model,  model_data = stock.create_prophet_model(days=30)

        return
