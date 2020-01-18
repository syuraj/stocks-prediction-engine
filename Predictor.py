import pandas as pd
from environs import Env
from datetime import date
import json
from mongo_connection import get_db_connection
from stock_trainer import StockTrainer

env = Env()
api_key = env('alphavantage_api_key')


class Predictor():

    def predict(self):
        symbol = 'TSLA'

        print(f'Running prediction for {symbol}')

        trainer = StockTrainer(api_key, "TSLA")
        model, stock_history, stock_forecast = trainer.create_prophet_model(30)

        self.save_model_in_db(symbol, stock_history[["ds", "Adj. Close"]].to_json(), stock_forecast[['ds', 'yhat']].to_json())

        return model,  stock_history, stock_forecast

    def save_model_in_db(self, symbol, stock_history, stock_forecast):
        conn = get_db_connection()
        stocksdb = conn["stocksdb"]
        modelCollection = stocksdb["models"]

        dbModel = {
            "symbol": symbol,
            "stock_history": stock_history,
            "stock_forecast": stock_forecast,
            "date_created": date.today().strftime("%Y-%m-%d")
        }

        modelCollection.find_one_and_replace({"symbol": symbol}, dbModel, upsert=True)
