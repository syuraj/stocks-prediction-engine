# Ref: https://towardsdatascience.com/stock-prediction-in-python-b66555171a2

# %%
from datetime import date
from Predictor import Predictor
from stocker import Stocker
from environs import Env
env = Env()
api_key = env('alphavantage_api_key')

# %%
model,  stock_history, stock_forecast = Predictor().predict()

# %%
print(stock_forecast.head(40))

# %%
# changepoint priors is the list of changepoints to evaluate
# stock.changepoint_prior_analysis(changepoint_priors=[0.001, 0.05, 0.1, 0.2])
stock_history[["ds", "Adj. Close"]].to_json()

# %%
# stock.changepoint_prior_validation(start_date='2016-01-04', end_date='2017-01-03',
#                                    changepoint_priors=[0.001, 0.05, 0.1, 0.2])

# %%
# trends, related_queries = stock.retrieve_google_trends("netflix", date_range=['2018-01-04', '2019-01-03'])
