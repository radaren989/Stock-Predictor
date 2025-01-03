import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense
from keras.layers import Flatten

# Step 1: Download stock data using yfinance
ticker = 'AAPL'  # Example: Apple stock
start_date = '2010-01-01'
end_date = '2023-01-01'

data = yf.download(ticker, start=start_date, end=end_date)

# Step 2: Preprocess the data
# We'll use 'Close' price for prediction
stock_data = data['Close'].values
stock_data = stock_data.reshape(-1, 1)

# Scale the data to range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
stock_data_scaled = scaler.fit_transform(stock_data)

# Step 3: Prepare the data for LSTM
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

time_step = 60
X, y = create_dataset(stock_data_scaled, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Step 4: Build the model (CNN for feature extraction + LSTM)
def create_model(input_shape):
    model = Sequential()

    # CNN layer for feature extraction
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))

    # LSTM layer
    model.add(LSTM(units=50, return_sequences=False))
    
    # Dense layer
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

model = create_model((X.shape[1], 1))

# Step 5: Train the model
model.fit(X, y, epochs=15, batch_size=32)

# Step 6: Predict the stock price
predicted_stock_price = model.predict(X)

# Step 7: Inverse transform to get the actual predicted stock prices
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
actual_stock_price = scaler.inverse_transform(y.reshape(-1, 1))

# Step 8: Plot the results
plt.figure(figsize=(14, 7))
plt.plot(actual_stock_price, color='blue', label='Actual Stock Price')
plt.plot(predicted_stock_price, color='red', label='Predicted Stock Price')
plt.title(f'{ticker} Stock Price Prediction with CNN + LSTM')
plt.xlabel('Time')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()

