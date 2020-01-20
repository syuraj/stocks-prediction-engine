
# %%
from lstm.LSTM_model import build_model_and_predict
from datetime import date
from prophet_predictor import ProphetPredictor
from stocker import Stocker
from environs import Env
import sys
sys.path.insert(0, "../")

env = Env()
api_key = env('alphavantage_api_key')


model_loss, X_predict = build_model_and_predict("AAPL")

print(X_predict * 200)
print(model_loss)
