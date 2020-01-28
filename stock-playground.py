
# %%
import sys
sys.path.insert(0, '../')
from lstm.LSTM_predictor import lstm_predict
from prophet.prophet_predictor import prophet_predict
from predictors import Predictor
from environs import Env

# env = Env()
# api_key = env('alphavantage_api_key')
model_loss, X_predict = lstm_predict('TSLA')

# stock_history, prophet_forecast, lstm_forecast = Predictor().predict()

# print(X_predict * 200)
# print(model_loss)

# %%
# import json
# X_predict
