import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from scipy.fft import fft

def calculate_ema(prices, span):
    return prices.ewm(span=span, adjust=False).mean()

def apply_fft(prices):
    fft_vals = fft(prices)
    return np.vstack((np.real(fft_vals), np.imag(fft_vals))).T

def preprocess_data(data, value_col, time_step):
    # Calculate EMA
    data['EMA'] = calculate_ema(data[value_col], span=10)
    data.dropna(inplace=True)
    
    # Extract prices and EMA values
    prices = data[value_col].values.reshape(-1, 1)
    ema_prices = data['EMA'].values.reshape(-1, 1)
    
    # Apply FFT
    fft_features = apply_fft(prices.flatten())
    
    # Scale the data
    scaler_prices = MinMaxScaler(feature_range=(0, 1))
    scaler_ema_prices = MinMaxScaler(feature_range=(0, 1))
    scaler_fft_features = MinMaxScaler(feature_range=(0, 1))
    
    scaled_prices = scaler_prices.fit_transform(prices)
    scaled_ema_prices = scaler_ema_prices.fit_transform(ema_prices)
    scaled_fft_features = scaler_fft_features.fit_transform(fft_features)
    
    # Combine features
    scaled_features = np.hstack((scaled_prices, scaled_ema_prices, scaled_fft_features))
    
    return scaled_features, scaler_prices

def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])
        Y.append(data[i + time_step, 0])  # Predicting the first feature (price)
    return np.array(X), np.array(Y)

def train_model(data, value_col, time_step, epochs=10, batch_size=32):
    # Preprocess data
    scaled_features, scaler = preprocess_data(data, value_col, time_step)
    
    # Split data into training and testing sets
    train_size = int(len(scaled_features) * 0.8)
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
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform predictions and actual values to original scale
    predictions_values = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test_values = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate MAE
    mae = mean_absolute_error(y_test_values, predictions_values)
    
    return predictions_values, y_test_values, mae

# Example usage with generic dataset
if __name__ == "__main__":
    # Generate a sample dataset
    dates = pd.date_range(start="2020-01-01", periods=100)
    values = np.linspace(100, 200, 100) + np.random.normal(0, 5, 100)
    data = pd.DataFrame({'date': dates, 'value': values})
    data.set_index('date', inplace=True)
    
    # Train the model
    time_step = 10
    predictions, actual, mae = train_model(data, 'value', time_step)
    
    # Print results
    print("Predicted Prices:", predictions.flatten())
    print("Actual Prices:", actual.flatten())
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
