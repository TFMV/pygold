import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D
from keras.losses import mean_squared_error
from keras.optimizers import Adam
import xgboost as xgb

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

# Reshape input data to be [samples, time steps, features] for CNN-LSTM
X_train_cnnlstm = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test_cnnlstm = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Build the CNN-LSTM model
model_cnnlstm = Sequential()
model_cnnlstm.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(time_step, X_train_cnnlstm.shape[2])))
model_cnnlstm.add(MaxPooling1D(pool_size=2))
model_cnnlstm.add(LSTM(units=50, return_sequences=True))
model_cnnlstm.add(Dropout(0.2))
model_cnnlstm.add(LSTM(units=50))
model_cnnlstm.add(Dropout(0.2))
model_cnnlstm.add(Dense(units=1))

# Compile the CNN-LSTM model
model_cnnlstm.compile(loss=mean_squared_error, optimizer=Adam(learning_rate=0.001))

# Train the CNN-LSTM model
model_cnnlstm.fit(X_train_cnnlstm, y_train, epochs=10, batch_size=32, validation_data=(X_test_cnnlstm, y_test), verbose=1)

# Reshape input data to be 2D for XGBoost (samples, features)
X_train_xgb = X_train.reshape(X_train.shape[0], -1)
X_test_xgb = X_test.reshape(X_test.shape[0], -1)

# Train the XGBoost model
model_xgb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, subsample=0.8, colsample_bytree=0.8, objective='reg:squarederror')
model_xgb.fit(X_train_xgb, y_train)

# Make predictions with both models
predictions_cnnlstm = model_cnnlstm.predict(X_test_cnnlstm)
predictions_xgb = model_xgb.predict(X_test_xgb)

# Combine predictions by averaging
combined_predictions = (predictions_cnnlstm.flatten() + predictions_xgb) / 2

# Inverse transform predictions and actual values to original scale
combined_predictions = scaler.inverse_transform(combined_predictions.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate MAE
mae = mean_absolute_error(y_test, combined_predictions)

# Print results
print("Predicted Prices:", combined_predictions.flatten())
print("Actual Prices:", y_test.flatten())
print(f"Mean Absolute Error (MAE): {mae:.4f}")
