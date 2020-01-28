import pandas as pd
from environs import Env
from datetime import date
import json
from prophet.prophet_stock_trainer import StockTrainer
import os

env = Env()
api_key = env('alphavantage_api_key')


def prophet_predict(symbol):
    trainer = StockTrainer(api_key, symbol)
    _, stock_history, stock_forecast = trainer.create_prophet_model(30)
    train_mean_error, test_mean_error = trainer.evaluate_prediction()

    return stock_forecast[['ds', 'yhat']].to_json(), stock_history[["ds", "Adj. Close"]].to_json(), train_mean_error, test_mean_error
