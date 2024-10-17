import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Step 1: Downloading Stock Data
start = datetime(2012, 1, 1)
end = datetime(2017, 12, 31)
ticker = ['AAPL', 'BIDU', 'MSFT', 'TXN']

# Download stock data
datas = yf.download(ticker, start=start, end=end)
datas = datas['Close']  # We only need the 'Close' price for this analysis

# Step 2: Define indicator functions

def EMA(df, n):
    return df.ewm(span=n, min_periods=n).mean()

def MOM(df, n):
    return df.diff(n)

def RSI(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Step 3: Calculating Technical Indicators for each stock
for t in ticker:
    datas[('EMA21', t)] = EMA(datas[t], 21)
    datas[('EMA63', t)] = EMA(datas[t], 63)
    datas[('RSI21', t)] = RSI(datas[t], 21)
    datas[('MOM21', t)] = MOM(datas[t], 21)

# Handle missing data (fill NaNs)
datas.fillna(method='bfill', inplace=True)

# Step 4: Generate Buy/Sell Signals using ML (Random Forest Classifier)

def generate_labels(df, short_ema, long_ema):
    """Generate buy (1), sell (-1), and hold (0) labels based on EMA crossover."""
    labels = np.zeros(df.shape[0])
    labels[df[short_ema] > df[long_ema]] = 1  # Buy signal
    labels[df[short_ema] < df[long_ema]] = -1  # Sell signal
    return labels

# Generate labels for each stock (using EMA crossovers)
for t in ticker:
    datas[('Signal', t)] = generate_labels(datas, ('EMA21', t), ('EMA63', t))

# Prepare features and labels for ML model
def prepare_ml_data(stock_data, ticker):
    """Prepares features and labels for ML models."""
    features = stock_data[[('EMA21', ticker), ('EMA63', ticker), ('RSI21', ticker), ('MOM21', ticker)]]
    labels = stock_data[('Signal', ticker)]
    return features, labels

# Step 5: Machine Learning Model (Random Forest for Buy/Sell Prediction)
def run_classification_model(ticker):
    features, labels = prepare_ml_data(datas, ticker)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{ticker} Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy:.2f}")

    return clf

# Run classification for each stock
models = {}
for t in ticker:
    models[t] = run_classification_model(t)

# Step 6: Predict Future Stock Prices using ML (Linear Regression)

def prepare_regression_data(stock_data, ticker):
    """Prepares features and labels for the regression model."""
    features = stock_data[[('EMA21', ticker), ('EMA63', ticker), ('RSI21', ticker), ('MOM21', ticker)]]
    labels = stock_data[ticker].shift(-1)  # Predict next day's close price
    return features[:-1], labels[:-1]

def run_regression_model(ticker):
    features, labels = prepare_regression_data(datas, ticker)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Train Linear Regression model
    reg = LinearRegression()
    reg.fit(X_train, y_train)

    # Make predictions
    y_pred = reg.predict(X_test)

    # Evaluate the model (R-squared score)
    r_squared = reg.score(X_test, y_test)
    print(f"\n{ticker} Regression Model R-squared: {r_squared:.2f}")

    return reg

# Run regression model for each stock
regression_models = {}
for t in ticker:
    regression_models[t] = run_regression_model(t)

# Step 7: Visualize Predictions
def plot_classification_results(ticker):
    # Plot stock price along with classification results (Buy/Sell signals)
    features, labels = prepare_ml_data(datas, ticker)
    clf = models[ticker]

    # Predictions for the entire dataset
    y_pred_full = clf.predict(features)

    plt.figure(figsize=(14, 8))
    plt.plot(datas.index, datas[ticker], label=f'{ticker} Closing Price', color='black')

    # Buy signals
    plt.plot(datas.index[y_pred_full == 1], datas[ticker][y_pred_full == 1], '^', markersize=10, color='g', label='Buy Signal')

    # Sell signals
    plt.plot(datas.index[y_pred_full == -1], datas[ticker][y_pred_full == -1], 'v', markersize=10, color='r', label='Sell Signal')

    plt.title(f"{ticker} - Random Forest Buy and Sell Predictions")
    plt.legend()
    plt.grid()
    plt.show()

for t in ticker:
    plot_classification_results(t)
