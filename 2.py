import scipy as sp
import math
import numpy as np
import pandas as pd
import requests
import sys
from keys import token
import statistics as stat


def request_data(ticker):
    ticker = ticker
    iex_call = f"https://iexapis.com/stable/stock/{ticker}/quote/"
    data = requests.get(iex_call).json()
    if data.status_code == 400:
        print("Encountered error")
        sys.exit(1)
    return data


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def split(stock_list):
    tickers = list(chunks(stock_list['Ticker'], 100))
    tkr_str = []
    for i in range(0, len(tickers)):
        tkr_str.append(','.join(tickers[i]))
    return tkr_str


def aggregate(tkr_str):
    data_pull = ['Ticker', 'Price', 'Price-to-Earnings Ratio', 'Number of Shares to Buy']
    data_agr = pd.DataFrame(columns=data_pull)

    for tkr_lst in tkr_str:
        #     print(symbol_strings)
        batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch/?types=quote&symbols={tkr_lst}&token={token}'
        data = requests.get(batch_api_call_url).json()
        for symbol in tkr_lst.split(','):
            data_agr = data_agr.append(pd.Series([symbol, data[symbol]['quote']['latestPrice'], data[symbol]['quote']['peRatio'], 'N/A'], index=data_pull), ignore_index=True)
    return data_agr


def pre_porc(data_agr):
    data_agr.sort_values('Price-to-Earnings Ratio', inplace=True)
    data_agr = data_agr[data_agr['Price-to-Earnings Ratio'] > 0]
    data_agr = data_agr[:50]
    data_agr.reset_index(inplace=True)
    data_agr.drop('index', axis=1, inplace=True)


def naive_strategy(data_agr, no_of_stocks):
    position_size = float(no_of_stocks) / len(no_of_stocks.index)
    for i in range(0, len(data_agr['Ticker'])):
        data_agr.loc[i, 'Number of Shares to Buy'] = math.floor(position_size / data_agr['Price'][i])
    return data_agr


def better_strat(tkr_str, data_agr):
    indicator_lst = ['Ticker','Price','Number of Shares to Buy','Price-to-Earnings Ratio','PE Percentile','Price-to-Book Ratio','PB Percentile','Price-to-Sales Ratio','PS Percentile','EV/EBITDA','EV/EBITDA Percentile','EV/GP','EV/GP Percentile','RV Score']
    data_agr = pd.DataFrame(columns=indicator_lst)

    for tkr_lst in tkr_str:
        batch_api_call_url = f'https://iexapis.com/stable/stock/market/batch?symbols={tkr_lst}&types=quote,advanced-stats&token={token}'
        data = requests.get(batch_api_call_url).json()
        for tkr in tkr_lst.split(','):
            enterprise_value = data[tkr]['advanced-stats']['enterpriseValue']
            ebitda = data[tkr]['advanced-stats']['EBITDA']
            gross_profit = data[tkr]['advanced-stats']['grossProfit']

            try:
                ev_to_ebitda = enterprise_value / ebitda
            except TypeError:
                ev_to_ebitda = np.NaN

            try:
                ev_to_gross_profit = enterprise_value / gross_profit
            except TypeError:
                ev_to_gross_profit = np.NaN

            data_agr = data_agr.append(
                pd.Series([tkr,data[tkr]['quote']['latestPrice'],'N/A',data[tkr]['quote']['peRatio'],'N/A',data[tkr]['advanced-stats']['priceToBook'],'N/A',data[tkr]['advanced-stats']['priceToSales'],'N/A',ev_to_ebitda,'N/A',ev_to_gross_profit,'N/A','N/A'],index=indicator_lst),ignore_index=True)

    for indicator in ['Price-to-Earnings Ratio', 'Price-to-Book Ratio', 'Price-to-Sales Ratio', 'EV/EBITDA', 'EV/GP']:
        data_agr[indicator].fillna(data_agr[indicator].mean(), inplace=True)
        metrics = {'Price-to-Earnings Ratio': 'PE Percentile','Price-to-Book Ratio': 'PB Percentile','Price-to-Sales Ratio': 'PS Percentile','EV/EBITDA': 'EV/EBITDA Percentile','EV/GP': 'EV/GP Percentile'}

        for row in data_agr.index:
            for metric in metrics.keys():
                data_agr.loc[row, metrics[metric]] = stats.percentileofscore(data_agr[metric], data_agr.loc[row, metric]) / 100

        # Print each percentile score to make sure it was calculated properly
        #for metric in metrics.values():
            #print(data_agr[metric])

        # Print the entire DataFrame
    return data_agr


def score(data_agr):
    perform_ind = {'Price-to-Earnings Ratio': 'PE Percentile','Price-to-Book Ratio': 'PB Percentile','Price-to-Sales Ratio': 'PS Percentile','EV/EBITDA': 'EV/EBITDA Percentile','EV/GP': 'EV/GP Percentile'}
    for record in data_agr.index:
        value_percentiles = []
        for metric in perform_ind.keys():
            value_percentiles.append(data_agr.loc[record, perform_ind[metric]])
        data_agr.loc[record, 'RV Score'] = stat.mean(value_percentiles)

    return data_agr


def strat(data_agr, stocks_to_buy):
    position_size = float(stocks_to_buy) / len(data_agr.index)
    for i in range(0, len(data_agr['Ticker']) - 1):
        data_agr.loc[i, 'Number of Shares to Buy'] = math.floor(position_size / data_agr['Price'][i])
    return data_agr


if __name__ == "__main__":
    Stock_list = pd.read_csv("SP500.csv")

