import sys
sys.path.insert(0, '../')
import pandas as pd
from environs import Env
from datetime import date
import json
from utilities.mongo_connection import get_db_connection
from prophet.prophet_predictor import prophet_predict
from lstm.LSTM_predictor import lstm_predict

import os

env = Env()
api_key = env('alphavantage_api_key')


class Predictor():

    def predict(self):
        symbol = 'TSLA'
        print(f'Running prediction for {symbol}')

        prophet_forecast, stock_history, train_mean_error, test_mean_error = prophet_predict(
            symbol)

        model_loss, lstm_forecast = lstm_predict(symbol)

        title = 'Stock Prediction for {} '.format(symbol)

        self.save_model_in_db(symbol, title, stock_history,
                              prophet_forecast, lstm_forecast, train_mean_error, test_mean_error)

        return stock_history, prophet_forecast, lstm_forecast

    def save_model_in_db(self, symbol, title, stock_history, prophet_forecast, lstm_forecast, train_mean_error, test_mean_error):
        conn = get_db_connection()
        stocksdb = conn["stocksdb"]
        modelCollection = stocksdb["models"]

        dbModel = {
            "symbol": symbol,
            "title": title,
            "stock_history": stock_history,
            "prophet_forecast": prophet_forecast,
            "lstm_forecast": lstm_forecast,
            "train_mean_error": train_mean_error,
            "test_mean_error": test_mean_error,
            "date_created": date.today().strftime("%Y-%m-%d")
        }

        modelCollection.find_one_and_replace(
            {"symbol": symbol}, dbModel, upsert=True)
