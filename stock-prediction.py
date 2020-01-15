# Ref: https://towardsdatascience.com/stock-prediction-in-python-b66555171a2

# %%
from Predictor import Predictor
from stocker import Stocker
from environs import Env
env = Env()
api_key = env('alphavantage_api_key')

# %%
stock = Stocker(api_key, 'TSLA')

# %%
stock.plot_stock(start_date='2019-10-1')

# %%
# predict days into the future
model, model_data = stock.create_prophet_model(days=30)

# %%
model,  model_data = Predictor.predict()

# %%



# %%
# stock.evaluate_prediction()


# %%
# changepoint priors is the list of changepoints to evaluate
# stock.changepoint_prior_analysis(changepoint_priors=[0.001, 0.05, 0.1, 0.2])

# %%
# stock.changepoint_prior_validation(start_date='2016-01-04', end_date='2017-01-03',
#                                    changepoint_priors=[0.001, 0.05, 0.1, 0.2])

# %%
# trends, related_queries = stock.retrieve_google_trends("netflix", date_range=['2018-01-04', '2019-01-03'])
