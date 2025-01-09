import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
import os

class StockPricePredictor:
    def __init__(self):
        self.time_step = 60
        self.data = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    def download_data(self, ticker, start_date, end_date):
        """Download stock data using yfinance."""
        self.data = yf.download(ticker, start=start_date, end=end_date)
        self.data = self.data['Close'].values.reshape(-1, 1)

    def preprocess_data(self):
        """Scale data to the range [0, 1]."""
        self.data = self.scaler.fit_transform(self.data)

    def create_dataset(self):
        """Create input-output sequences for training."""
        X, y = [], []
        for i in range(self.time_step, len(self.data)):
            X.append(self.data[i - self.time_step:i, 0])
            y.append(self.data[i, 0])
        X, y = np.array(X), np.array(y)
        return X.reshape(X.shape[0], X.shape[1], 1), y

    def build_model(self, input_shape):
        """Build the CNN + LSTM model."""
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def train_model(self, X, y, epochs=15, batch_size=32):
        """Train the model on the provided dataset."""
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def load_model(self, model_path):
        """Load a pre-compiled and saved model."""
        if os.path.exists(model_path):
            self.model = load_model(model_path)
        else:
            raise FileNotFoundError(f"Model file {model_path} not found.")

    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)

    def inverse_transform(self, data):
        """Inverse transform scaled data to original values."""
        return self.scaler.inverse_transform(data)

    def plot_results(self, actual, predicted, ticker):
        """Plot the actual vs predicted stock prices."""
        plt.figure(figsize=(14, 7))
        plt.plot(actual, color='blue', label='Actual Stock Price')
        plt.plot(predicted, color='red', label='Predicted Stock Price')
        plt.title(f'{ticker} Stock Price Prediction with CNN + LSTM')
        plt.xlabel('Time')
        plt.ylabel('Stock Price (USD)')
        plt.legend()
        plt.show()

""" 
# Training script example
if __name__ == "__main__":
    predictor = StockPricePredictor()

    # Parameters
    ticker = 'AAPL'
    start_date = '2010-01-01'
    end_date = '2023-01-01'
    model_path = 'stock_predictor_model.h5'

    # Check if the model exists, train if not
    if not os.path.exists(model_path):
        predictor.download_data(ticker, start_date, end_date)
        predictor.preprocess_data()
        X, y = predictor.create_dataset()
        predictor.build_model(input_shape=(X.shape[1], 1))
        predictor.train_model(X, y, epochs=15, batch_size=32)
        predictor.model.save(model_path)
    else:
        predictor.load_model(model_path)

    # Example prediction (using the same data for simplicity)
    predictor.download_data(ticker, start_date, end_date)
    predictor.preprocess_data()
    X, y = predictor.create_dataset()
    predicted_stock_price = predictor.predict(X)
    predicted_stock_price = predictor.inverse_transform(predicted_stock_price)
    actual_stock_price = predictor.inverse_transform(y.reshape(-1, 1))

    # Plot results
    predictor.plot_results(actual_stock_price, predicted_stock_price, ticker)
"""
