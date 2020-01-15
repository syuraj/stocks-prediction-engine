from environs import Env
from stocker import Stocker
import pandas as pd
from mongo_connection import get_db_connection
import json

env = Env()
api_key = env('alphavantage_api_key')


class Predictor():

    @classmethod
    def predict(self):
        symbol = 'TSLA'
        stock = Stocker(api_key,  symbol)

        print(f'Running prediction for {symbol}')

        model,  model_data = stock.create_prophet_model(days=30)

        self.save_model_in_db(symbol, model_data[['ds', 'yhat']].to_json())

        return model,  model_data

    @classmethod
    def save_model_in_db(self, symbol, model_data):
        conn = get_db_connection()
        stocksdb = conn["stocksdb"]
        modelCollection = stocksdb["models"]

        model = {"symbol": symbol, "model": model_data}

        modelCollection.insert_one(model)
