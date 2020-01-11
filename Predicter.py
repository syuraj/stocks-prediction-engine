from ConfigReader import ConfigReader
from stocker import Stocker
import pandas as pd

configReader = ConfigReader()
api_key = configReader.readConfig('alphavantage.api_key')


class Predicter():

    @staticmethod
    def predict():
        stock = Stocker(api_key,  'TSLA')
        stock.plot_stock(start_date=pd.to_datetime('2018-10-1'))

        model,  model_data = stock.create_prophet_model(days=30)

        return
