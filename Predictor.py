from environs import Env
import pandas as pd
from mongo_connection import get_db_connection
import json
from stock_trainer import StockTrainer

env = Env()
api_key = env('alphavantage_api_key')


class Predictor():

    def predict(self):
        symbol = 'TSLA'

        print(f'Running prediction for {symbol}')

        trainer = StockTrainer(api_key, "TSLA")
        model, future = trainer.create_prophet_model()

        self.save_model_in_db(symbol, future[['ds', 'yhat']].to_json())

        return model,  future

    def save_model_in_db(self, symbol, model_data):
        conn = get_db_connection()
        stocksdb = conn["stocksdb"]
        modelCollection = stocksdb["models"]

        model = {"symbol": symbol, "model": model_data}

        modelCollection.insert_one(model)
