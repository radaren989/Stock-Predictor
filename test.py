import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Download stock data using yfinance
ticker = "AAPL"  # You can change this to any stock symbol
data = yf.download(ticker, start="2010-01-01", end="2023-01-01")

# Step 2: Prepare the data
# Use the 'Close' prices for prediction
data = data[['Close']]

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create the dataset for LSTM (using past 60 days to predict next day's price)
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Reshape data to fit LSTM model
X, y = create_dataset(scaled_data)

# Reshape X to be 3D as required by LSTM (samples, time steps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Step 3: Create the LSTM Model
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Prediction of next day's price
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = create_model((X.shape[1], 1))

# Step 4: Train the Model with EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X, y, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

# Step 5: Make Predictions
predicted_stock_price = model.predict(X)

# Step 6: Reverse scaling to get the actual stock prices
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
actual_stock_price = scaler.inverse_transform(y.reshape(-1, 1))

# Step 7: Plot the results
plt.figure(figsize=(14, 7))
plt.plot(actual_stock_price, color='blue', label='Actual Stock Price')
plt.plot(predicted_stock_price, color='red', label='Predicted Stock Price')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()

