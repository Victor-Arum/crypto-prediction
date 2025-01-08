import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def data_dictionary(data):
    data_dict = {}
    data_dict["All Coins"] = data
    # Loop through each unique cryptocurrency
    for crypto in data['Crypto'].unique():
        subset = data[data['Crypto'] == crypto]
        data_dict[crypto] = subset

    return data_dict


def download_data(cryptos, duration):
    # Define the date range for the past year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(duration-1))

    # Initialize an empty DataFrame to hold all the data
    all_data = pd.DataFrame()

    # Fetch data for each cryptocurrency
    for crypto in cryptos:
        data = yf.download(crypto, start=start_date, end=end_date)
        data['Crypto'] = crypto
        all_data = pd.concat([all_data, data])

    # Reset index to flatten the DataFrame
    all_data.reset_index(inplace=True)

    # Sort by 'Crypto' and reset index
    data = all_data.sort_values(by=['Crypto', 'Date']).reset_index(drop=True)

    return data

def clean_data(data):
    # Drop any rows with missing values
    data.dropna(inplace=True)
    return data




# List of 30 cryptocurrencies
crypto_list = [
            'BTC-USD','ETH-USD','ICP-USD','DOGE-USD','SOL-USD',
            'USDT-USD','USDC-USD','BNB-USD','XRP-USD','ADA-USD',
            'DAI-USD','WTRX-USD', 'DOT-USD','HEX-USD','TRX-USD',
            'SHIB-USD','LEO-USD','WBTC-USD','AVAX-USD','EGLD-USD',
            'MATIC-USD','UNI-USD','STETH-USD','LTC-USD','FTT-USD',
            'CRO-USD', 'FIL-USD', 'RUNE-USD', 'XMR-USD', 'ETC-USD'
]

