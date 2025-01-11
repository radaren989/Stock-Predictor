from flask import Blueprint, render_template, request
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import os
import matplotlib
from sklearn.preprocessing import StandardScaler
import warnings
import datetime as dt
import matplotlib.dates as mdates
import math
import random

# For visualization
from mplfinance.original_flavor import candlestick_ohlc

# Ignore warnings
warnings.filterwarnings("ignore")

# For data import
import kagglehub

# Downloading datasets
dgawlik_nyse_path = kagglehub.dataset_download('dgawlik/nyse')
borismarjanovic_price_volume_data_for_all_us_stocks_etfs_path = kagglehub.dataset_download(
    'borismarjanovic/price-volume-data-for-all-us-stocks-etfs')
camnugent_sandp500_path = kagglehub.dataset_download('camnugent/sandp500')

mattiuzc_stock_exchange_data_path = kagglehub.dataset_download('mattiuzc/stock-exchange-data')
bryanb_cac40_stocks_dataset_path = kagglehub.dataset_download('bryanb/cac40-stocks-dataset')

print('Data source import complete.')

# Set matplotlib to use non-interactive Agg backend
matplotlib.use('Agg')

# Blueprint for routes
app_routes = Blueprint('app_routes', __name__)
def save_plot(plt, plot_name):
    # Save the plot to a file,

    plot_path = os.path.join("static", "plots", plot_name)


    plt.savefig(plot_path)
    plt.close()  # Close the plot to prevent it from being displayed interactively
    return plot_path


@app_routes.route('/', methods=['GET', 'POST'])
def index():
    company_name = None  # Initialize company_name to None to avoid UnboundLocalError
    if request.method == 'POST':
        company_name = request.form.get('company_name')  # Get the company name from the form



    # Load dataset
        df = pd.read_csv(
            "C:\\Users\\Excalibur\\.cache\\kagglehub\\datasets\\bryanb\\cac40-stocks-dataset\\versions\\18\\preprocessed_CAC40.csv", parse_dates=['Date'])
        df.drop(['Unnamed: 0'], axis=1, inplace=True)

        # Define function to get data for a specific
        def specific_data(company, start, end):
            company_data = df[df['Name'] == company]
            date_filtered_data = company_data[(company_data['Date'] > start) & (company_data['Date'] < end)]
            return date_filtered_data

        start_date = dt.datetime(2010, 1, 1)
        end_date = dt.datetime(2025, 1, 10)

        # Get the specific company data
        specific_df = specific_data(company_name, start_date, end_date)

        # Visualization
        specific_df['Date'] = pd.to_datetime(specific_df['Date'])

        # Line chart for closing prices over time
        plt.figure(figsize=(15, 6))
        plt.plot(specific_df['Date'], specific_df['Closing_Price'], marker='.')
        plt.title('Closing Prices Over Time')
        plt.xlabel('Date')
        plt.ylabel('Closing Price')
        plt.xticks(rotation=45)
        plt.grid(True)
        plot_url_1 = save_plot(plt, 'closing_prices.png')

        # Prepare for candlestick chart
        matplotlib_date = mdates.date2num(specific_df['Date'])
        ohlc = np.vstack((matplotlib_date, specific_df['Open'], specific_df['Daily_High'], specific_df['Daily_Low'],
                          specific_df['Closing_Price'])).T

        plt.figure(figsize=(15, 6))
        ax = plt.subplot()
        candlestick_ohlc(ax, ohlc, width=0.6, colorup='g', colordown='r')
        ax.xaxis_date()
        plt.title('Candlestick Chart')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.grid(True)
        plot_url_2 = save_plot(plt, 'candlestick_chart.png')

        # Closing Prices and Moving Average plot
        window = 30
        plt.figure(figsize=(15, 6))
        plt.plot(specific_df['Date'], specific_df['Closing_Price'], label='Closing Price', linewidth=2)
        plt.plot(specific_df['Date'], specific_df['Closing_Price'].rolling(window=window).mean(),
                 label=f'{window}-Day Moving Avg', linestyle='--')
        plt.title(f'Closing Prices and {window}-Day Moving Average')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plot_url_3 = save_plot(plt, 'moving_average.png')

        # Monthly Seasonality
        specific_df['Month'] = specific_df['Date'].dt.month
        monthly_average = specific_df.groupby('Month')['Closing_Price'].mean()

        plt.figure(figsize=(15, 6))
        plt.plot(monthly_average.index, monthly_average.values, marker='o')
        plt.title(f'Monthly Seasonality of {company_name}')
        plt.xlabel('Months')
        plt.ylabel('Average Closing Price')
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.grid(True)
        plot_url_4 = save_plot(plt, 'monthly_seasonality.png')

        # Prepare data for training
        new_df = specific_df.reset_index()['Closing_Price']
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(np.array(new_df).reshape(-1, 1))

        # Split data into training and testing sets
        train_size = int(len(scaled_data) * 0.8)
        train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

        n_past = 60
        X_train, y_train = [], []
        for i in range(n_past, len(train_data)):
            X_train.append(train_data[i - n_past:i, 0])
            y_train.append(train_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)

        X_test, y_test = [], []
        for i in range(n_past, len(test_data)):
            X_test.append(test_data[i - n_past:i, 0])
            y_test.append(test_data[i, 0])
        X_test, y_test = np.array(X_test), np.array(y_test)

        # Reshape data for LSTM model
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Model training (LSTM)
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(loss='mean_squared_error', optimizer='adam')

        # Train the model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32, verbose=1)

        # Predictions
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Inverse transformation of predictions
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)

        look_back = 60

        # Initialize an array for plotting the train predictions
        trainPredictPlot = np.empty_like(new_df)
        trainPredictPlot[:] = np.nan
        # Assign the predicted values to the appropriate location for train predictions
        trainPredictPlot[look_back:len(train_predict) + look_back] = train_predict.flatten()

        # Initialize an array for plotting the test predictions
        testPredictPlot = np.empty_like(new_df)
        testPredictPlot[:] = np.nan
        # Calculate the starting index for the test predictions
        test_start = len(new_df) - len(test_predict)
        # Assign the predicted values to the appropriate location for test predictions
        testPredictPlot[test_start:] = test_predict.flatten()

        # Plotting predicted vs actual prices


        # Predict next 10 days
        last_sequence = X_test[-1]
        last_sequence = last_sequence.reshape(1, n_past, 1)

        future_predictions = []
        for _ in range(10):
            next_day_prediction = model.predict(last_sequence)
            future_predictions.append(scaler.inverse_transform(next_day_prediction)[0, 0])
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = next_day_prediction

        # Convert future predictions to string for rendering
        future_predictions_str = "\n".join([f"Day {i + 1}: {pred:.2f}" for i, pred in enumerate(future_predictions)])

        plt.figure(figsize=(15, 6))
        plt.plot(scaler.inverse_transform(scaled_data), color='black', label=f"Actual {company_name} price")
        plt.plot(trainPredictPlot, color='red', label=f"Predicted {company_name} price(train set)")
        plt.plot(testPredictPlot, color='blue', label=f"Predicted {company_name} price(test set)")
        plt.title(f"{company_name} Share Price")
        plt.xlabel("Time")
        plt.ylabel("Share Price")
        plt.legend()
        plt.grid(True)
        plot_url_5 = save_plot(plt, 'predicted_vs_actual_prices.png')  # Save with a unique name

        # Plotting predicted stock price for the next 10 days
        plt.figure(figsize=(15, 6))
        plt.plot(future_predictions, marker='*', color='b')  # predictions_next_10_days
        plt.title(f'Predicted stock price of {company_name} for next 10 days')
        plt.xlabel('Days')
        plt.ylabel('Price')
        plt.xticks(range(0, 10), ['Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7', 'Day8', 'Day9', 'Day10'])
        plt.grid(True)
        plot_url_6 = save_plot(plt, 'predicted_next_10_days.png')  # Save with a unique name
        plt.show()

        return render_template('index.html',
                               company_name=company_name,
                               plot_url_1=plot_url_1,
                               plot_url_2=plot_url_2,
                               plot_url_3=plot_url_3,
                               plot_url_4=plot_url_4,
                               # Make sure this points to the correct plot (for example, predicted vs actual)
                               plot_url_5=plot_url_5,  # New plot for predicted vs actual prices
                               plot_url_6=plot_url_6,  # New plot for predicted next 10 days
                               future_predictions=future_predictions_str)

    return render_template('index.html', company_name=company_name)