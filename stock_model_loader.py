import pandas as pd
import numpy as np
import fbprophet
import pytrends
from pytrends.request import TrendReq
from alpha_vantage.timeseries import TimeSeries


class StockModelLoader():
    def __init__(self, api_key, ticker):
        ticker = ticker.upper()

        self.api_key = api_key
        self.ticker = ticker

    def load(self):
        stock_file_path = './data/' + self.ticker + '.csv'

        try:
            stock = pd.read_csv(stock_file_path, index_col=0, parse_dates=['index'])
        except FileNotFoundError:
            try:
                ts = TimeSeries(key=self.api_key, output_format='pandas', indexing_type='integer')
                stock, meta_data = ts.get_daily_adjusted(symbol=self.ticker, outputsize='full')

                stock.rename(columns={"index": "Date"},  inplace=True)
                stock.to_csv(stock_file_path)
            except Exception as e:
                print('Error Retrieving Data.')
                print(e)
                return

        # Set the index to a column called Date
        stock = stock.reset_index(level=0)
        stock['Date'] = pd.to_datetime(stock['Date'])

        # Columns required for prophet
        stock['ds'] = stock['Date']

        if ('Adj. Close' not in stock.columns):
            stock['Adj. Close'] = stock['5. adjusted close']
            stock['Adj. Open'] = stock['1. open'] * stock['5. adjusted close'] / stock['4. close']
            stock['Adj. Volume'] = stock['6. volume'] * stock['4. close'] / stock['5. adjusted close']
            stock['Adj. High'] = stock['2. high'] * stock['5. adjusted close'] / stock['4. close']
            stock['Adj. Close'] = stock['4. close'] * stock['5. adjusted close'] / stock['4. close']

        stock['y'] = stock['Adj. Close']
        stock['Daily Change'] = stock['Adj. Close'] - stock['Adj. Open']
        stock.drop(['1. open', '2. high', '3. low', '4. close', '5. adjusted close',
                    '6. volume', '7. dividend amount', '8. split coefficient'], axis=1)

        # Minimum and maximum date in range
        self.min_date = min(stock['Date'])
        self.max_date = max(stock['Date'])
        self.min_date = pd.to_datetime(self.min_date)
        self.max_date = pd.to_datetime(self.max_date)

        # Find max and min prices and dates on which they occurred
        self.max_price = np.max(stock['y'])
        self.min_price = np.min(stock['y'])

        self.min_price_date = stock[stock['y'] == self.min_price]['Date']
        self.min_price_date = self.min_price_date[self.min_price_date.index[0]]
        self.max_price_date = stock[stock['y'] == self.max_price]['Date']
        self.max_price_date = self.max_price_date[self.max_price_date.index[0]]

        # The starting price (starting with the opening price)
        self.starting_price = float(stock.ix[0, 'Adj. Open'])

        # The most recent price
        self.most_recent_price = float(stock.ix[len(stock) - 1, 'y'])

        # Whether or not to round dates
        self.round_dates = True

        # Number of years of data to train on
        self.training_years = 3

        # Prophet parameters
        # Default prior from library
        self.changepoint_prior_scale = 0.05
        self.weekly_seasonality = False
        self.daily_seasonality = False
        self.monthly_seasonality = True
        self.yearly_seasonality = True
        self.changepoints = None

        return stock
