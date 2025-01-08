from sklearn.svm import SVR
from keras.models import Sequential
from keras.layers import Dense, LSTM
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')
from tensorflow import keras
from math import sqrt
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go




predict_with_prophet_data = None
predict_with_prophet_cache_status = None

@st.cache_data
def load_predict_with_prophet(cryptos, dates):
    global predict_with_prophet_cache_status
    predict_with_prophet_cache_status = "Fetching new data..."

    data = predict_with_prophet_data.copy()
    data.reset_index(inplace=True)

    models_and_plots = {}

    for crypto in data['Crypto'].unique():
        crypto_data = data[data['Crypto'] == crypto]

        # Prepare the data
        df = crypto_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])

        # Train-Test split (80% train, 20% test)
        split_index = int(len(df) * 0.2)
        train_data = df[:-split_index]
        test_data = df[-split_index:]

        # Fit the Prophet model on training data
        model = Prophet()
        model.fit(train_data)

        # Make predictions for the test data period
        future = model.make_future_dataframe(periods=len(test_data), freq='D')
        forecast = model.predict(future)

        # Extract the relevant predictions for test data
        predictions = forecast[['ds', 'yhat']].set_index('ds').iloc[-len(test_data):]

        # Calculate error metrics
        mse = mean_squared_error(test_data['y'], predictions['yhat'])
        mae = mean_absolute_error(test_data['y'], predictions['yhat'])
        rmse = np.sqrt(mse)

        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }

        # Create an interactive Plotly plot
        fig = go.Figure()

        # Add traces for train data, test data, and predictions
        fig.add_trace(go.Scatter(x=train_data['ds'], y=train_data['y'], mode='lines', name='Train Data', line=dict(color='brown')))
        fig.add_trace(go.Scatter(x=test_data['ds'], y=test_data['y'], mode='lines', name='Test Data', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=predictions.index, y=predictions['yhat'], mode='lines', name='Predictions', line=dict(color='red')))

        # Customize the layout
        fig.update_layout(
            title=f'{crypto} _ Prophet Train, Test, and Predicted Prices',
            xaxis_title='Date',
            yaxis_title='Close Price',
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            legend=dict(x=0, y=1.0),
            margin=dict(l=40, r=40, t=40, b=40),
            height=500,
        )

        # Store model and plot
        models_and_plots[crypto] = {
            'prophet_pred_plot': fig,
            'metrics': metrics,
            'predictions': predictions['yhat']
        }

    return models_and_plots


def predict_with_prophet(data, show_plot):
    global predict_with_prophet_data
    predict_with_prophet_data = data

    global predict_with_prophet_cache_status
    predict_with_prophet_cache_status = None

    # Number of unique cryptocurrencies
    cryptos = sorted(data['Crypto'].unique())
    dates = sorted(data['Date'].unique())

    models_and_plots = load_predict_with_prophet(cryptos, dates)

    if show_plot:
        # Check cache status
        if predict_with_prophet_cache_status == "Fetching new data...":
            st.success("Success!!!")
        else:
            st.info("Success!!!")

        # Display the plots for each cryptocurrency
        for crypto, content in models_and_plots.items():
            st.subheader(f'{crypto} _ Prophet Train, Test, and Predicted Prices')

            # Display the Plotly plot
            st.plotly_chart(content['prophet_pred_plot'], use_container_width=True)

            st.subheader(f"{crypto} _ Prophet Error Metrics:")
            for key, value in content['metrics'].items():
                st.write(f"Calculated {key} is {value:.4f}")

    return models_and_plots





forecast_with_prophet_data = None
forecast_with_prophet_cache_status = None

@st.cache_data
def load_forecast_with_prophet(cryptos, dates, duration):
    global forecast_with_prophet_cache_status
    forecast_with_prophet_cache_status = "Fetching new data..."

    data = forecast_with_prophet_data.copy()
    data.reset_index(inplace=True)

    models_and_plots = {}

    for crypto in data['Crypto'].unique():
        crypto_data = data[data['Crypto'] == crypto]

        # Prepare the data for Prophet
        df = crypto_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        df['ds'] = pd.to_datetime(df['ds'])

        # Fit the Prophet model
        model = Prophet()
        model.fit(df)

        # Make predictions for the specified future duration
        future = model.make_future_dataframe(periods=duration, freq='D')
        forecast = model.predict(future)

        # Extract the relevant predictions
        predictions = forecast[['ds', 'yhat']].set_index('ds').iloc[-duration:]

        # Create a Plotly figure for the original and forecasted values
        fig_forecast = go.Figure()

        # Plot original data
        fig_forecast.add_trace(go.Scatter(
            x=df['ds'], y=df['y'], mode='lines', name='Original Data', line=dict(color='blue')
        ))

        # Plot forecasted prices
        fig_forecast.add_trace(go.Scatter(
            x=predictions.index, y=predictions['yhat'], mode='lines', name='Forecasted Prices', line=dict(color='red')
        ))

        fig_forecast.update_layout(
            title=f'{crypto} - Prophet Original and Forecasted Prices',
            xaxis_title='Date',
            yaxis_title='Close Price',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
        )

        # Store the forecast plot and metrics
        models_and_plots[crypto] = {
            'prophet_forecast_plot': fig_forecast,
            'future_dates': predictions.index,
            'predictions': predictions['yhat']
        }

    return models_and_plots


def forecast_with_prophet(data, duration, show_plot):
    global forecast_with_prophet_data
    forecast_with_prophet_data = data

    global forecast_with_prophet_cache_status
    forecast_with_prophet_cache_status = None

    # Number of unique cryptocurrencies
    cryptos = sorted(data['Crypto'].unique())
    dates = sorted(data['Date'].unique())

    models_and_plots = load_forecast_with_prophet(cryptos, dates, duration)

    if show_plot:
        # Check cache status
        if forecast_with_prophet_cache_status == "Fetching new data...":
            st.success("Success!!!")
        else:
            st.info("Success!!!")

        if models_and_plots:
            # Display the plots for each cryptocurrency
            for crypto, content in models_and_plots.items():
                st.subheader(f'{crypto} - Prophet Original and Forecasted Prices')

                # Display Plotly chart for Prophet forecast
                st.plotly_chart(content['prophet_forecast_plot'], use_container_width=True)

                # Create and display a dataframe for current and forecasted prices
                subset1 = data[data['Crypto'] == crypto]

                current_price_dict = {
                    "Crypto": [crypto],
                    "Date": [subset1["Date"].values[-1]],
                    "Price Class": ["Current Price"],
                    "Price": [subset1["Close"].values[-1]]
                }
                current_price_df = pd.DataFrame(current_price_dict)

                forecasted_price_dict = {
                    "Crypto": [crypto],
                    "Date": [content["future_dates"][-1]],
                    "Price Class": ["Forecasted Price"],
                    "Price": [content["predictions"][-1]]
                }
                forecasted_price_df = pd.DataFrame(forecasted_price_dict)

                # Combine and display current and forecasted prices
                current_price_df = pd.concat([current_price_df, forecasted_price_df])
                current_price_df.reset_index(inplace=True, drop=True)

                st.subheader(f'{crypto} - Prophet Current and Forecasted Prices')
                crypto_price = current_price_df["Price"]
                st.write(f"The current price of {crypto} is {crypto_price.iloc[0]:.4f}")
                st.write(f"The future price of {crypto} is {crypto_price.iloc[1]:.4f}")

    return models_and_plots



def rolling_range_mean_std(series, window, start, end):
    """Compute rolling mean and std for a specific range within the window."""
    means = []
    stds = []

    for i in range(len(series)):
        if i >= window - 1:
            window_slice = series[max(0, i - window + 1):i + 1]
            if len(window_slice) >= end:
                subset = window_slice[start:end]
                means.append(subset.mean())
                stds.append(subset.std())
            else:
                means.append(None)
                stds.append(None)
        else:
            means.append(None)
            stds.append(None)

    return pd.Series(means, index=series.index), pd.Series(stds, index=series.index)


def add_features(df):
    # Add basic lag features
    df['lag_1'] = df['Close'].shift(1)
    df['lag_7'] = df['Close'].shift(7)
    df['lag_30'] = df['Close'].shift(15)
    df['lag_45'] = df['Close'].shift(30)
    df['lag_60'] = df['Close'].shift(60)

    # Add rolling window statistics
    df['rolling_mean_7'], df['rolling_std_7'] = rolling_range_mean_std(df['Close'], 8, 0, 8)
    # df['rolling_std_7'] = df['Close'].rolling(window=7).std()
    df['rolling_mean_30'], df['rolling_std_30'] = rolling_range_mean_std(df['Close'], 16, 0, 8)
    # df['rolling_std_30'] = df['Close'].rolling(window=30).std()
    df['rolling_mean_45'], df['rolling_std_45'] = rolling_range_mean_std(df['Close'], 31, 0, 15)
    # df['rolling_std_45'] = df['Close'].rolling(window=45).std()
    df['rolling_mean_60'], df['rolling_std_60'] = rolling_range_mean_std(df['Close'], 61, 0, 30)
    # df['rolling_std_60'] = df['Close'].rolling(window=60).std()

    # Add time-based features
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    # Drop rows with missing values created by lag/rolling
    df.dropna(inplace=True)
    return df





predict_with_svm_data = None
predict_with_svm_cache_status = None

@st.cache_data
def load_predict_with_svm(cryptos, dates):
    global predict_with_svm_cache_status
    predict_with_svm_cache_status = "Fetching new data..."

    data = predict_with_svm_data.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    models_and_plots = {}

    for crypto in data['Crypto'].unique():
        subset1 = data[data['Crypto'] == crypto]
        crypto_data = add_features(subset1.copy())  # Add features

        # Train-test split (80% train, 20% test)
        split_index = int(len(subset1) * 0.2)
        train_data = crypto_data[:-split_index]
        test_data = crypto_data[-split_index:]

        train_X = train_data.drop(columns=['Close', 'Crypto', 'Open', 'High', 'Low', 'Adj Close', 'Volume'])
        train_y = train_data['Close']

        test_X = test_data.drop(columns=['Close', 'Crypto', 'Open', 'High', 'Low', 'Adj Close', 'Volume'])
        test_y = test_data['Close']

        # Scale the features (SVM is sensitive to scaling)
        scaler = RobustScaler()
        train_X = scaler.fit_transform(train_X)
        test_X = scaler.transform(test_X)

        # Define and train the SVM model
        svr_rbf = SVR(kernel='rbf', C=1e5, gamma=0.001)  # Adjust hyperparameters if needed
        svr_rbf.fit(train_X, train_y)

        # Predict
        predictions = svr_rbf.predict(test_X)

        # Calculate error metrics
        mse = mean_squared_error(test_y, predictions)
        mae = mean_absolute_error(test_y, predictions)
        rmse = np.sqrt(mse)

        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }

        # Create an interactive Plotly plot
        fig = go.Figure()

        # Add traces for train data, test data, and predictions
        fig.add_trace(go.Scatter(x=train_data.index, y=train_data['Close'], mode='lines', name='Train Data', line=dict(color='brown')))
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data['Close'], mode='lines', name='Test Data', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=test_data.index, y=predictions, mode='lines', name='Predictions', line=dict(color='red')))

        # Customize the layout
        fig.update_layout(
            title=f'{crypto} - SVM Train, Test, and Predicted Prices',
            xaxis_title='Date',
            yaxis_title='Close Price',
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            legend=dict(x=0, y=1.0),
            margin=dict(l=40, r=40, t=40, b=40),
            height=500,
        )

        # Store model and plot
        models_and_plots[crypto] = {
            'svm_pred_plot': fig,
            'metrics': metrics,
            'predictions': predictions
        }

    return models_and_plots


def predict_with_svm(data, show_plot):
    global predict_with_svm_data
    predict_with_svm_data = data

    global predict_with_svm_cache_status
    predict_with_svm_cache_status = None

    # Number of unique cryptocurrencies
    cryptos = sorted(data['Crypto'].unique())
    dates = sorted(data['Date'].unique())

    models_and_plots = load_predict_with_svm(cryptos, dates)

    if show_plot:
        # Check cache status
        if predict_with_svm_cache_status == "Fetching new data...":
            st.success("Success!!!")
        else:
            st.info("Success!!!")

        # Display the plots for each cryptocurrency
        for crypto, content in models_and_plots.items():
            st.subheader(f'{crypto} - SVM Train, Test, and Predicted Prices')

            # Display the Plotly plot
            st.plotly_chart(content['svm_pred_plot'], use_container_width=True)

            st.subheader(f"{crypto} - SVM Error Metrics:")
            for key, value in content['metrics'].items():
                st.write(f"Calculated {key} is {value:.4f}")

    return models_and_plots






forecast_with_svm_data = None
forecast_with_svm_cache_status = None

@st.cache_data
def load_forecast_with_svm(cryptos, dates, duration):
    global forecast_with_svm_cache_status
    forecast_with_svm_cache_status = "Fetching new data..."

    data = forecast_with_svm_data.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    models_and_plots = {}

    for crypto in data['Crypto'].unique():
        crypto_data = data[data['Crypto'] == crypto]
        crypto_data = crypto_data.drop(columns=['Crypto', 'Open', 'High', 'Low', 'Adj Close', 'Volume'])
        crypto_data = add_features(crypto_data)  # Use the same feature engineering function

        # Train-test split (80% train)
        train_data, test_data = crypto_data[:], None
        train_X = train_data.drop(columns=['Close'])[:-1]
        train_y = train_data['Close'][1:]

        # Scale the features (SVM is sensitive to scaling)
        scaler = StandardScaler()
        train_X = scaler.fit_transform(train_X)

        # Train the SVM model
        svr_rbf = SVR(kernel='rbf', C=1e5, gamma=0.001)
        svr_rbf.fit(train_X, train_y)

        # Forecast future dates
        future_dates = pd.date_range(start=crypto_data.index[-1] + pd.Timedelta(days=1), periods=duration)

        train_and_future_data = crypto_data.copy()

        for i in range(duration):
            future_X = train_and_future_data.drop(columns=['Close'])[-1:]
            future_Xs = scaler.transform(future_X)
            pred = svr_rbf.predict(future_Xs)[0]

            # Add prediction to the future data
            new_row = pd.DataFrame({
                'Date': [future_dates[i]],
                'Close': [pred],
                'lag_1': [0], 'lag_7': [0], 'lag_30': [0], 'lag_45': [0], 'lag_60': [0],
                'rolling_mean_7': [0], 'rolling_std_7': [0],
                'rolling_mean_30': [0], 'rolling_std_30': [0],
                'rolling_mean_45': [0], 'rolling_std_45': [0],
                'rolling_mean_60': [0], 'rolling_std_60': [0],
                'day_of_week': [0], 'month': [0]
            })
            new_row.set_index('Date', inplace=True)
            train_and_future_data = pd.concat([train_and_future_data, new_row])

            # Update lags and rolling features for the next iteration
            train_and_future_data['lag_1'].iloc[-1] = train_and_future_data['Close'].iloc[-2]
            train_and_future_data['lag_7'].iloc[-1] = train_and_future_data['Close'].iloc[-8]
            train_and_future_data['lag_30'].iloc[-1] = train_and_future_data['Close'].iloc[-16]
            train_and_future_data['lag_45'].iloc[-1] = train_and_future_data['Close'].iloc[-31]
            train_and_future_data['lag_60'].iloc[-1] = train_and_future_data['Close'].iloc[-61]
            train_and_future_data['rolling_mean_7'].iloc[-1] = train_and_future_data['Close'].iloc[-8:].mean()
            train_and_future_data['rolling_std_7'].iloc[-1] = train_and_future_data['Close'].iloc[-8:].std()
            train_and_future_data['rolling_mean_30'].iloc[-1] = train_and_future_data['Close'].iloc[-16:-8].mean()
            train_and_future_data['rolling_std_30'].iloc[-1] = train_and_future_data['Close'].iloc[-16:-8].std()
            train_and_future_data['rolling_mean_45'].iloc[-1] = train_and_future_data['Close'].iloc[-31:-16].mean()
            train_and_future_data['rolling_std_45'].iloc[-1] = train_and_future_data['Close'].iloc[-31:-16].std()
            train_and_future_data['rolling_mean_60'].iloc[-1] = train_and_future_data['Close'].iloc[-61:-31].mean()
            train_and_future_data['rolling_std_60'].iloc[-1] = train_and_future_data['Close'].iloc[-61:-31].std()
            train_and_future_data['day_of_week'].iloc[-1] = train_and_future_data.index[-1].dayofweek
            train_and_future_data['month'].iloc[-1] = train_and_future_data.index[-1].month

        predictions = train_and_future_data["Close"].iloc[-duration:]

        # Create an interactive Plotly figure
        fig_forecast = go.Figure()

        # Plot original data
        fig_forecast.add_trace(go.Scatter(
            x=train_data.index, y=train_data['Close'], mode='lines', name='Original Data', line=dict(color='blue')
        ))

        # Plot forecasted prices
        fig_forecast.add_trace(go.Scatter(
            x=future_dates, y=predictions, mode='lines', name='Forecasted Prices', line=dict(color='red')
        ))

        fig_forecast.update_layout(
            title=f'{crypto} - SVM Original and Forecasted Prices',
            xaxis_title='Date',
            yaxis_title='Close Price',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
        )

        # Store the forecast plot and predictions
        models_and_plots[crypto] = {
            'svm_forecast_plot': fig_forecast,
            'future_dates': future_dates,
            'predictions': predictions
        }

    return models_and_plots


def forecast_with_svm(data, duration, show_plot):
    global forecast_with_svm_data
    forecast_with_svm_data = data

    global forecast_with_svm_cache_status
    forecast_with_svm_cache_status = None

    # Number of unique cryptocurrencies
    cryptos = sorted(data['Crypto'].unique())
    dates = sorted(data['Date'].unique())

    models_and_plots = load_forecast_with_svm(cryptos, dates, duration)

    if show_plot:
        # Check cache status
        if forecast_with_svm_cache_status == "Fetching new data...":
            st.success("Success!!!")
        else:
            st.info("Success!!!")

        if models_and_plots:
            # Display the plots for each cryptocurrency
            for crypto, content in models_and_plots.items():

                st.subheader(f'{crypto} - SVM Original and Forecasted Prices')
                # Display Plotly chart for SVM forecast
                st.plotly_chart(content['svm_forecast_plot'], use_container_width=True)

                subset1 = data[data['Crypto'] == crypto]

                # Display current and forecasted prices
                current_price_dict = {
                    "Crypto": [crypto],
                    "Date": [subset1.index[-1]],
                    "Price Class": ["Current Price"],
                    "Price": [subset1["Close"].values[-1]]
                }
                current_price_df = pd.DataFrame(current_price_dict)

                forecasted_price_dict = {
                    "Crypto": [crypto],
                    "Date": [content["future_dates"][-1]],
                    "Price Class": ["Forecasted Price"],
                    "Price": [content["predictions"][-1]]
                }
                forecasted_price_df = pd.DataFrame(forecasted_price_dict)

                # Combine and display current and forecasted prices
                current_price_df = pd.concat([current_price_df, forecasted_price_df])
                current_price_df.reset_index(inplace=True, drop=True)

                st.subheader(f'{crypto} - SVM Current and Forecasted Prices')
                crypto_price = current_price_df["Price"]
                st.write(f"The current price of {crypto} is {crypto_price.iloc[0]:.4f}")
                st.write(f"The future price of {crypto} is {crypto_price.iloc[1]:.4f}")

    return models_and_plots






predict_with_arima_data = None
predict_with_arima_cache_status = None

@st.cache_data
def load_predict_with_arima(cryptos, dates):
    """
    Train ARIMA model with 80% of crypto data and test the model with the remaining 20%.
    """
    global predict_with_arima_cache_status
    predict_with_arima_cache_status = "Fetching new data..."

    data = predict_with_arima_data.copy()
    # Ensure the 'Date' column is set as the index
    data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' column to datetime if not already
    data.set_index('Date', inplace=True)

    models_and_plots = {}

    for crypto in data['Crypto'].unique():
        crypto_data = data[data['Crypto'] == crypto]

        # Split the data into training and testing
        split_index = int(len(crypto_data) * 0.2)
        train_data = crypto_data['Close'][:-split_index]
        test_data = crypto_data['Close'][-split_index:]

        # Set ARIMA orders (you can change these based on tuning)
        order = (1, 0, 0)
        seasonal_order = (2, 1, 0, 28)

        model = SARIMAX(train_data,
                        order=order,
                        seasonal_order=seasonal_order)

        result = model.fit()

        start = len(train_data)
        end = len(train_data) + len(test_data) - 1

        predictions = result.predict(start, end, typ='levels').rename("Predictions")

        # Calculate error metrics
        mse = mean_squared_error(test_data, predictions)
        mae = mean_absolute_error(test_data, predictions)
        rmse = sqrt(mse)

        # Save metrics to dictionary
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }

        # Create an interactive Plotly plot
        fig = go.Figure()

        # Add traces for train data, test data, and predictions
        fig.add_trace(go.Scatter(x=train_data.index, y=train_data, mode='lines', name='Train Data', line=dict(color='brown')))
        fig.add_trace(go.Scatter(x=test_data.index, y=test_data, mode='lines', name='Test Data', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=test_data.index, y=predictions, mode='lines', name='Predictions', line=dict(color='red')))

        # Customize the layout
        fig.update_layout(
            title=f'{crypto} - ARIMA Train, Test, and Predicted Prices',
            xaxis_title='Date',
            yaxis_title='Close Price',
            plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
            legend=dict(x=0, y=1.0),
            margin=dict(l=40, r=40, t=40, b=40),
            height=500,
        )

        # Store model and plot
        models_and_plots[crypto] = {
            'arima_pred_plot': fig,
            'metrics': metrics,
            'predictions': predictions
        }

    return models_and_plots


def predict_with_arima(data, show_plot):
    global predict_with_arima_data
    predict_with_arima_data = data

    global predict_with_arima_cache_status
    predict_with_arima_cache_status = None

    # Number of unique cryptocurrencies
    cryptos = sorted(data['Crypto'].unique())
    dates = sorted(data['Date'].unique())

    models_and_plots = load_predict_with_arima(cryptos, dates)

    if show_plot:
        # Check cache status
        if predict_with_arima_cache_status == "Fetching new data...":
            st.success("Success!!!")
        else:
            st.info("Success!!!")

        # Display the plots for each cryptocurrency
        for crypto, content in models_and_plots.items():
            st.subheader(f'{crypto} - ARIMA Train, Test, and Predicted Prices')

            # Display the Plotly plot
            st.plotly_chart(content['arima_pred_plot'], use_container_width=True)

            st.subheader(f"{crypto} - ARIMA Error Metrics:")
            for key, value in content['metrics'].items():
                st.write(f"Calculated {key} is {value:.4f}")

    return models_and_plots





forecast_with_arima_data = None
forecast_with_arima_cache_status = None

@st.cache_data
def load_forecast_with_arima(cryptos, dates, duration):
    """
    Train ARIMA model with 80% of crypto data and test the model with the remaining 20%.
    """
    global forecast_with_arima_cache_status
    forecast_with_arima_cache_status = "Fetching new data..."

    data = forecast_with_arima_data.copy()
    # Ensure the 'Date' column is set as the index
    data['Date'] = pd.to_datetime(data['Date'])  # Convert 'Date' column to datetime if not already
    data.set_index('Date', inplace=True)

    models_and_plots = {}

    for crypto in data['Crypto'].unique():
        crypto_data = data[data['Crypto'] == crypto]

        full_data = crypto_data['Close']

        # ARIMA model parameters
        order = (1, 0, 0)
        seasonal_order = (2, 1, 0, 28)

        model = SARIMAX(full_data,
                        order=order,
                        seasonal_order=seasonal_order)

        result = model.fit()

        # Forecast
        start = len(full_data)
        end = len(full_data) + duration - 1

        predictions = result.predict(start, end, typ='levels').rename("Forecast")

        # Get the last date from the data
        last_date = full_data.index[-1]

        # Generate a range of future dates starting from the last date
        future_dates = pd.date_range(start=last_date, periods=duration + 1, freq='D')[1:]

        # Create Plotly figure for ARIMA forecasts
        fig_forecast = go.Figure()

        # Plot original data
        fig_forecast.add_trace(go.Scatter(x=full_data.index, y=full_data, mode='lines', name='Original Data', line=dict(color='blue')))

        # Plot forecasted data
        fig_forecast.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines', name='Forecasted Data', line=dict(color='red')))

        fig_forecast.update_layout(
            title=f'{crypto} - ARIMA Original and Forecasted Prices',
            xaxis_title='Date',
            yaxis_title='Close Price',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
        )

        # Store model and plot data
        models_and_plots[crypto] = {
            'arima_fore_plot': fig_forecast,
            'future_dates': future_dates,
            'predictions': predictions
        }

    return models_and_plots


def forecast_with_arima(data, duration, show_plot):
    global forecast_with_arima_data
    forecast_with_arima_data = data

    global forecast_with_arima_cache_status
    forecast_with_arima_cache_status = None

    # Number of unique cryptocurrencies
    cryptos = sorted(data['Crypto'].unique())
    dates = sorted(data['Date'].unique())

    models_and_plots = load_forecast_with_arima(cryptos, dates, duration)

    if show_plot:
        # Check cache status
        if forecast_with_arima_cache_status == "Fetching new data...":
            st.success("Success!!!")
        else:
            st.info("Success!!!")

        # Display the plots for each cryptocurrency
        for crypto, content in models_and_plots.items():
            st.subheader(f'{crypto} - ARIMA Original and Forecasted Prices')

            # Display the Plotly chart for the original and forecasted data
            st.plotly_chart(content['arima_fore_plot'], use_container_width=True)

            subset1 = data[data['Crypto'] == crypto]

            # Create DataFrame for current and forecasted prices
            current_price_dict = {
                "Crypto": [crypto],
                "Date": [subset1.index[-1]],
                "Price Class": ["Current Price"],
                "Price": [subset1["Close"].values[-1]]
            }
            current_price_df = pd.DataFrame(current_price_dict)

            forecasted_price_dict = {
                "Crypto": [crypto],
                "Date": [content["future_dates"][-1]],
                "Price Class": ["Forecasted Price"],
                "Price": [content["predictions"][-1]]
            }
            forecasted_price_df = pd.DataFrame(forecasted_price_dict)

            # Combine current and forecasted prices into one DataFrame
            current_price_df = pd.concat([current_price_df, forecasted_price_df])
            current_price_df.reset_index(inplace=True, drop=True)

            st.subheader(f'{crypto} - ARIMA Current and Forecasted Prices')
            crypto_price = current_price_df["Price"]
            st.write(f"The current price of {crypto} is {crypto_price.iloc[0]:.4f}")
            st.write(f"The future price of {crypto} is {crypto_price.iloc[1]:.4f}")

    return models_and_plots




def lstm_create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)





predict_with_lstm_data = None
predict_with_lstm_cache_status = None

@st.cache_data
def load_predict_with_lstm(cryptos, dates):
    global predict_with_lstm_cache_status
    predict_with_lstm_cache_status = "Fetching new data..."

    data = predict_with_lstm_data.copy()

    time_steps = 30
    n_features = 1

    models_and_plots = {}

    for crypto in data['Crypto'].unique():
        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')

        subset1 = data[data['Crypto'] == crypto]
        subset2 = subset1["Close"].values.reshape(-1, 1)

        # Scaling the data
        scaler = RobustScaler()
        subset2_scaled = scaler.fit_transform(subset2)

        # Prepare sequences with optimal time_steps
        X, y = lstm_create_sequences(subset2_scaled, time_steps)

        # Split into train and test sets
        split_index = int(len(subset1) * 0.2)
        X_train, X_test = X[:-split_index], X[-split_index:]
        y_train, y_test = y[:-split_index], y[-split_index:]

        input_shape = (time_steps, n_features)
        final_model = keras.Sequential()
        final_model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128, input_shape=input_shape)))
        final_model.add(keras.layers.Dropout(rate=0.2))
        final_model.add(keras.layers.Dense(units=1))
        final_model.compile(loss='mean_squared_error', optimizer='adam')

        history = final_model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=32,
            validation_data=(X_test, y_test),
            shuffle=False,
            callbacks=[earlyStopping]
        )

        # Make predictions on the test set
        y_pred = final_model.predict(X_test)

        # Inverse transform the predictions and actual values to the original scale
        y_train_inverse = scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_test_inverse = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_inverse = scaler.inverse_transform(y_pred).flatten()

        # Get the corresponding dates for y_train and y_test
        y_train_dates = subset1["Date"][time_steps:].values[:-split_index]
        y_test_dates = subset1["Date"][time_steps:].values[-split_index:]

        # Calculate error metrics
        mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
        mse = mean_squared_error(y_test_inverse, y_pred_inverse)
        rmse = sqrt(mse)

        # Save metrics to dictionary
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }

        # Create a Plotly plot for y_train, y_test, and y_pred on the original scale
        fig_pred = go.Figure()

        fig_pred.add_trace(go.Scatter(x=y_train_dates, y=y_train_inverse, mode='lines', name='Train', line=dict(color='brown')))
        fig_pred.add_trace(go.Scatter(x=y_test_dates, y=y_test_inverse, mode='lines', name='Test', line=dict(color='blue')))
        fig_pred.add_trace(go.Scatter(x=y_test_dates, y=y_pred_inverse, mode='lines', name='Predicted', line=dict(color='red')))

        fig_pred.update_layout(
            title=f'{crypto} - LSTM Train, Test, and Predicted Prices',
            xaxis_title='Date',
            yaxis_title='Close Price',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
        )

        # Create a Plotly plot for history loss and validation loss
        fig_hist = go.Figure()

        fig_hist.add_trace(go.Scatter(x=list(range(len(history.history['loss']))), y=history.history['loss'], mode='lines', name='Loss', line=dict(color='orange')))
        fig_hist.add_trace(go.Scatter(x=list(range(len(history.history['val_loss']))), y=history.history['val_loss'], mode='lines', name='Validation Loss', line=dict(color='purple')))

        fig_hist.update_layout(
            title=f'{crypto} - LSTM Model Training Loss and Validation Loss',
            xaxis_title='Epochs',
            yaxis_title='Loss',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
        )

        models_and_plots[crypto] = {
            'pred_plot': fig_pred,
            'history_plot': fig_hist,
            'metrics': metrics,
            'predictions': y_pred_inverse
        }

    return models_and_plots


def predict_with_lstm(data, show_plot):
    global predict_with_lstm_data
    predict_with_lstm_data = data

    global predict_with_lstm_cache_status
    predict_with_lstm_cache_status = None

    # Number of unique cryptocurrencies
    cryptos = sorted(data['Crypto'].unique())
    dates = sorted(data['Date'].unique())

    models_and_plots = load_predict_with_lstm(cryptos, dates)

    if show_plot:
        # Check cache status
        if predict_with_lstm_cache_status == "Fetching new data...":
            st.success("Success!!!")
        else:
            st.info("Success!!!")

        # Display the plots for each cryptocurrency
        for crypto, content in models_and_plots.items():
            st.subheader(f'{crypto} - LSTM Train, Test, and Predicted Prices')

            # Display the Plotly plot for predictions
            st.plotly_chart(content['pred_plot'], use_container_width=True)

            st.subheader(f"{crypto} - LSTM Error Metrics:")
            for key, value in content['metrics'].items():
                st.write(f"Calculated {key} is {value:.4f}")

    return models_and_plots






forecast_with_lstm_data = None
forecast_with_lstm_cache_status = None

@st.cache_data
def load_forecast_with_lstm(cryptos, dates, duration):
    global forecast_with_lstm_cache_status
    forecast_with_lstm_cache_status = "Fetching new data..."

    data = forecast_with_lstm_data.copy()

    models_and_plots = {}

    # Number of days to forecast
    n_days = duration
    time_steps = 30
    n_features = 1

    for crypto in data['Crypto'].unique():
        earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')

        subset1 = data[data['Crypto'] == crypto]
        subset2 = subset1["Close"].values.reshape(-1, 1)

        # Scaling the data only once here
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(subset2)

        # Start with the last available sequence from the training data
        initial_input = scaled_data[-time_steps:]

        X, y = lstm_create_sequences(scaled_data, time_steps)

        input_shape = (time_steps, n_features)
        final_model = keras.Sequential()
        final_model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128, input_shape=input_shape)))
        final_model.add(keras.layers.Dropout(rate=0.2))
        final_model.add(keras.layers.Dense(units=1))
        final_model.compile(loss='mean_squared_error', optimizer='adam')

        history = final_model.fit(
            X, y,
            epochs=30,
            batch_size=32,
            validation_split=0.1,
            shuffle=False,
            callbacks=[earlyStopping]
        )

        # Initialize a list to hold future predictions
        predictions = []

        # Generate future predictions
        for _ in range(n_days):
            # Reshape input to be compatible with LSTM model
            input_data = initial_input.reshape((1, time_steps, n_features))

            # Predict the next value
            next_pred = final_model.predict(input_data)

            # Append the prediction to the list
            predictions.append(next_pred[0])

            # Update the input sequence by appending the prediction and removing the oldest data point
            initial_input = np.append(initial_input[1:], next_pred, axis=0)

        # Inverse scale the predictions back to the original scale
        predictions = scaler.inverse_transform(predictions)

        # Get the last date from the data
        last_date = subset1["Date"].values[-1]

        # Generate a range of future dates starting from the last date
        future_dates = pd.date_range(start=last_date, periods=n_days + 1, freq='D')[1:]

        # Create a Plotly plot for original and forecasted data
        fig_forecast = go.Figure()

        # Plot original data
        fig_forecast.add_trace(go.Scatter(x=subset1["Date"], y=subset2.flatten(), mode='lines', name='Original Data', line=dict(color='blue')))

        # Plot forecasted data
        fig_forecast.add_trace(go.Scatter(x=future_dates, y=predictions.flatten(), mode='lines', name='Forecasted Data', line=dict(color='red')))

        fig_forecast.update_layout(
            title=f'{crypto} - LSTM Original and Forecasted Prices',
            xaxis_title='Date',
            yaxis_title='Close Price',
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
        )

        models_and_plots[crypto] = {
            'lstm_fore_plot': fig_forecast,
            'future_dates': future_dates,
            "predictions": predictions.flatten()
        }

    return models_and_plots


def forecast_with_lstm(data, duration, show_plot):
    global forecast_with_lstm_data
    forecast_with_lstm_data = data

    global forecast_with_lstm_cache_status
    forecast_with_lstm_cache_status = None

    # Number of unique cryptocurrencies
    cryptos = sorted(data['Crypto'].unique())
    dates = sorted(data['Date'].unique())

    models_and_plots = load_forecast_with_lstm(cryptos, dates, duration)

    if show_plot:
        # Check cache status
        if forecast_with_lstm_cache_status == "Fetching new data...":
            st.success("Success!!!")
        else:
            st.info("Success!!!")

        # Display the plots for each cryptocurrency
        for crypto, content in models_and_plots.items():
            st.subheader(f'{crypto} - LSTM Original and Forecasted Prices')

            # Display the Plotly chart for the original and forecasted data
            st.plotly_chart(content['lstm_fore_plot'], use_container_width=True)

            subset1 = data[data['Crypto'] == crypto]

            # Current and forecasted prices dataframe
            current_price_dict = {
                "Crypto": [crypto],
                "Date": [subset1["Date"].values[-1]],
                "Price Class": ["Current Price"],
                "Price": [subset1["Close"].values[-1]]
            }
            current_price_df = pd.DataFrame(current_price_dict)

            forecasted_price_dict = {
                "Crypto": [crypto],
                "Date": [content["future_dates"][-1]],
                "Price Class": ["Forecasted Price"],
                "Price": [content["predictions"][-1]]
            }
            forecasted_price_df = pd.DataFrame(forecasted_price_dict)

            current_price_df = pd.concat([current_price_df, forecasted_price_df])
            current_price_df.reset_index(inplace=True, drop=True)

            st.subheader(f'{crypto} - LSTM Current and Forecasted Prices')

            crypto_price = current_price_df["Price"]
            st.write(f"The current price of {crypto} is {crypto_price.iloc[0]:.4f}")
            st.write(f"The future price of {crypto} is {crypto_price.iloc[1]:.4f}")

    return models_and_plots



