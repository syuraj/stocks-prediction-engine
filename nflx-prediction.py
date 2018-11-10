# Ref: https://towardsdatascience.com/stock-prediction-in-python-b66555171a2
# %%
from ConfigReader import ConfigReader
configReader = ConfigReader()
api_key = configReader.readConfig('quandl.ApiConfig.api_key')

# %%
from stocker import Stocker
netflix = Stocker(api_key, 'NFLX')

# %%
netflix.plot_stock()

# %%
# predict days into the future
model, model_data = netflix.create_prophet_model(days=90)

# %%
netflix.evaluate_prediction()

# %%
# changepoint priors is the list of changepoints to evaluate
# netflix.changepoint_prior_analysis(changepoint_priors=[0.001, 0.05, 0.1, 0.2])

# %%
# netflix.changepoint_prior_validation(start_date='2016-01-04', end_date='2017-01-03',
#                                      changepoint_priors=[0.001, 0.05, 0.1, 0.2])

# %%
# trends, related_queries = netflix.retrieve_google_trends("netflix", date_range=['2018-01-04', '2019-01-03'])
