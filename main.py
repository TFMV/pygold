import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D
from keras.losses import mean_squared_error
from keras.optimizers import Adam

# Function to calculate EMA
def calculate_ema(prices, span):
    return prices.ewm(span=span, adjust=False).mean()

# Fetch gold futures data
gold_data = yf.download('GC=F', start='2000-01-01', end='2024-05-31')

# Calculate EMA with a span of 10 days (or any other period you find suitable)
gold_data['EMA'] = calculate_ema(gold_data['Close'], span=10)

# Drop NaN values that may have been introduced by the EMA calculation
gold_data.dropna(inplace=True)

# Preprocess the data
gold_prices = gold_data['Close'].values.reshape(-1, 1)
ema_prices = gold_data['EMA'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_gold_prices = scaler.fit_transform(gold_prices)
scaled_ema_prices = scaler.fit_transform(ema_prices)

# Combine gold prices and EMA as features
scaled_features = np.hstack((scaled_gold_prices, scaled_ema_prices))

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])
        Y.append(data[i + time_step, 0])  # Predicting the gold price
    return np.array(X), np.array(Y)

# Define time step
time_step = 100

# Split data into training and testing sets
train_size = int(len(scaled_features) * 0.8)
test_size = len(scaled_features) - train_size
train_data, test_data = scaled_features[:train_size, :], scaled_features[train_size:, :]

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input data to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Build the CNN-LSTM model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(time_step, X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the model
model.compile(loss=mean_squared_error, optimizer=Adam(learning_rate=0.001))

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Make predictions
predictions = model.predict(X_test)

# Inverse transform predictions and actual values to original scale
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate MAE
mae = mean_absolute_error(y_test, predictions)

# Print results
print("Predicted Prices:", predictions.flatten())
print("Actual Prices:", y_test.flatten())
print(f"Mean Absolute Error (MAE): {mae:.4f}")
