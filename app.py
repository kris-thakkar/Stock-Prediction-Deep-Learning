import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from datetime import datetime, timedelta

# Function to get stock data
def get_stock_data(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # Last 5 years
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to prepare data for LSTM
def prepare_data(df, feature='Close', look_back=60):
    data = df[feature].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X = []
    y = []

    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

# Function to build and train the LSTM model
def build_and_train_model(X_train, y_train, epochs=20, batch_size=32):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    return model

# Main function
def main():
    ticker = input("Enter the stock ticker symbol (e.g., AAPL for Apple Inc.): ").upper()
    print(f"Fetching data for {ticker}...")
    data = get_stock_data(ticker)

    if data.empty:
        print("Failed to retrieve data. Please check the ticker symbol and your internet connection.")
        return

    # Prepare the data
    look_back = 60
    X, y, scaler = prepare_data(data, look_back=look_back)

    # Split into training and testing datasets
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build and train the model
    print("Training the model...")
    model = build_and_train_model(X_train, y_train)

    # Predict for the next day
    last_60_days = data['Close'].values[-look_back:]
    last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
    X_input = np.reshape(last_60_days_scaled, (1, look_back, 1))
    predicted_price_scaled = model.predict(X_input)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)

    # Get today's closing price
    today_price = data['Close'].values[-1]

    # Decision logic
    if predicted_price > today_price:
        decision = "Buy"
    else:
        decision = "Sell"

    print(f"\nToday's Closing Price: ${today_price:.2f}")
    print(f"Predicted Tomorrow's Closing Price: ${predicted_price[0][0]:.2f}")
    print(f"Recommendation: {decision}")

    # Plotting the results
    plt.figure(figsize=(12,6))
    plt.plot(data['Close'], label='Historical Close Prices')
    plt.title(f'{ticker} Stock Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
