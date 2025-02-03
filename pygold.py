import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization, 
                                     GlobalAveragePooling1D, MultiHeadAttention, 
                                     Conv1D, MaxPooling1D, LSTM, Concatenate, Average)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
import optuna

# ---------------------------
# 1. Advanced Feature Engineering
# ---------------------------
def compute_features(data):
    # Basic price features
    data['log_price'] = np.log(data['value'])
    data['return'] = data['value'].pct_change()
    
    # Exponential Moving Average (EMA)
    data['EMA'] = data['value'].ewm(span=10, adjust=False).mean()
    
    # Volatility (rolling standard deviation)
    data['volatility'] = data['return'].rolling(window=10).std()
    
    # Wavelet transform features (placeholder: in practice, replace with actual coefficients)
    data['wavelet_feature'] = data['value']  # Replace with actual wavelet coefficients
    
    data.dropna(inplace=True)
    return data

# ---------------------------
# 2. Create a sliding window dataset
# ---------------------------
def create_dataset(data, feature_cols, time_step=10):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[feature_cols].iloc[i:i+time_step].values)
        y.append(data['value'].iloc[i+time_step])
    return np.array(X), np.array(y)

# ---------------------------
# 3. Custom Loss Function (Asymmetric Loss)
# ---------------------------
def custom_asymmetric_loss(y_true, y_pred):
    """
    Penalizes underestimation more than overestimation.
    For example, if the prediction is lower than the true value,
    multiply the squared error by a factor alpha.
    """
    alpha = 1.5
    diff = y_true - y_pred
    loss = tf.where(diff > 0, alpha * tf.square(diff), tf.square(diff))
    return tf.reduce_mean(loss)

# ---------------------------
# 4. Transformer Encoder Block
# ---------------------------
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs  # Residual connection

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res  # Second residual connection

# ---------------------------
# 5. Build the Ensemble Model (Transformer + CNN–LSTM)
# ---------------------------
def build_ensemble_model(input_shape, transformer_params, cnn_lstm_params, ensemble_method='weighted'):
    inputs = Input(shape=input_shape)
    
    # --- Transformer Branch ---
    x1 = inputs
    for _ in range(transformer_params['num_transformer_blocks']):
        x1 = transformer_encoder(x1, 
                                 head_size=transformer_params['head_size'], 
                                 num_heads=transformer_params['num_heads'], 
                                 ff_dim=transformer_params['ff_dim'], 
                                 dropout=transformer_params['dropout'])
    x1 = GlobalAveragePooling1D()(x1)
    for units in transformer_params['mlp_units']:
        x1 = Dense(units, activation="relu")(x1)
        x1 = Dropout(transformer_params['mlp_dropout'])(x1)
    transformer_output = Dense(1)(x1)
    
    # --- CNN-LSTM Branch (Updated) ---
    x2 = Conv1D(filters=cnn_lstm_params['filters'], 
                kernel_size=cnn_lstm_params['kernel_size'], 
                activation='relu',
                padding='same')(inputs)  # Added padding='same'
    
    # Only apply pooling if sequence length is sufficient
    if input_shape[0] >= 4:  # Ensure minimum sequence length
        x2 = MaxPooling1D(pool_size=2, padding='same')(x2)
    
    x2 = LSTM(cnn_lstm_params['lstm_units'], return_sequences=True)(x2)
    x2 = Dropout(cnn_lstm_params['dropout'])(x2)
    x2 = LSTM(cnn_lstm_params['lstm_units'])(x2)
    x2 = Dropout(cnn_lstm_params['dropout'])(x2)
    cnn_lstm_output = Dense(1)(x2)
    
    # --- Ensemble ---
    if ensemble_method == 'weighted':
        # Concatenate branch outputs and learn a meta representation
        combined = Concatenate()([transformer_output, cnn_lstm_output])
        ensemble_output = Dense(1)(combined)
    else:
        # Simple average of predictions
        ensemble_output = Average()([transformer_output, cnn_lstm_output])
    
    model = Model(inputs, ensemble_output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss=custom_asymmetric_loss)
    return model

# ---------------------------
# 6. Hyperparameter Tuning with Optuna
# ---------------------------
def objective(trial):
    # Updated hyperparameter ranges
    time_step = trial.suggest_int("time_step", 10, 30)  # Increased minimum time step
    epochs = trial.suggest_int("epochs", 30, 100)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # Transformer parameters with adjusted ranges
    transformer_params = {
        'head_size': trial.suggest_int("head_size", 16, 64),
        'num_heads': trial.suggest_int("num_heads", 2, 4),
        'ff_dim': trial.suggest_int("ff_dim", 32, 128),
        'num_transformer_blocks': trial.suggest_int("num_transformer_blocks", 1, 3),
        'mlp_units': [trial.suggest_int("mlp_units", 32, 128)],
        'dropout': trial.suggest_float("transformer_dropout", 0.1, 0.3),
        'mlp_dropout': trial.suggest_float("transformer_mlp_dropout", 0.1, 0.3)
    }
    
    # CNN-LSTM parameters with adjusted ranges
    cnn_lstm_params = {
        'filters': trial.suggest_int("filters", 16, 64),
        'kernel_size': trial.suggest_int("kernel_size", 2, 3),  # Reduced max kernel size
        'lstm_units': trial.suggest_int("lstm_units", 16, 64),
        'dropout': trial.suggest_float("cnn_lstm_dropout", 0.1, 0.3)
    }
    
    # Input validation
    if not isinstance(global_data, pd.DataFrame):
        raise ValueError("global_data must be a pandas DataFrame")
    
    # Use a copy of the global data and compute features
    try:
        data_proc = compute_features(global_data.copy())
    except Exception as e:
        print(f"Error in feature computation: {str(e)}")
        return float('inf')  # Return worst possible score
        
    feature_cols = ['value', 'EMA', 'volatility', 'wavelet_feature']
    
    # Validate feature columns
    missing_cols = [col for col in feature_cols if col not in data_proc.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Scale features robustly
    scaler = RobustScaler()
    data_proc[feature_cols] = scaler.fit_transform(data_proc[feature_cols])
    
    # Create dataset with validation
    try:
        X, y = create_dataset(data_proc, feature_cols, time_step)
        if len(X) < 100:  # Minimum dataset size check
            print("Warning: Dataset too small for reliable training")
            return float('inf')
    except Exception as e:
        print(f"Error in dataset creation: {str(e)}")
        return float('inf')
    
    # Split into training and validation sets (time–ordered split)
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    input_shape = X_train.shape[1:]  # (time_step, num_features)
    
    # Build the ensemble model with the current hyperparameters
    model = build_ensemble_model(input_shape, transformer_params, cnn_lstm_params, ensemble_method='weighted')
    
    # Use early stopping to prevent over–fitting
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model (use verbose=0 for tuning)
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val),
                        epochs=epochs, 
                        batch_size=batch_size, 
                        callbacks=[early_stopping],
                        verbose=0)
    
    # Evaluate on the validation set (using our custom loss)
    val_loss = model.evaluate(X_val, y_val, verbose=0)
    return val_loss

# ---------------------------
# 7. Final Training using Best Hyperparameters from Optuna
# ---------------------------
def train_final_model(best_params, data, epochs, batch_size, time_step):
    # Prepare data
    data_proc = compute_features(data.copy())
    feature_cols = ['value', 'EMA', 'volatility', 'wavelet_feature']
    scaler = RobustScaler()
    data_proc[feature_cols] = scaler.fit_transform(data_proc[feature_cols])
    X, y = create_dataset(data_proc, feature_cols, time_step)
    
    # Use an 80-20 split for training and testing
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Unpack hyperparameters from best_params
    transformer_params = {
        'head_size': best_params["head_size"],
        'num_heads': best_params["num_heads"],
        'ff_dim': best_params["ff_dim"],
        'num_transformer_blocks': best_params["num_transformer_blocks"],
        'mlp_units': [best_params["mlp_units"]],
        'dropout': best_params["transformer_dropout"],
        'mlp_dropout': best_params["transformer_mlp_dropout"]
    }
    cnn_lstm_params = {
        'filters': best_params["filters"],
        'kernel_size': best_params["kernel_size"],
        'lstm_units': best_params["lstm_units"],
        'dropout': best_params["cnn_lstm_dropout"]
    }
    
    input_shape = X_train.shape[1:]
    model = build_ensemble_model(input_shape, transformer_params, cnn_lstm_params, ensemble_method='weighted')
    model.summary()
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test),
                        epochs=epochs, 
                        batch_size=batch_size, 
                        callbacks=[early_stopping],
                        verbose=1)
    
    predictions = model.predict(X_test).flatten()
    mae = mean_absolute_error(y_test, predictions)
    print("Final Test MAE: {:.4f}".format(mae))
    return model, predictions, y_test, mae, scaler

# Add new utility function for model validation
def validate_model_inputs(X_train, y_train, X_val, y_val):
    """Validate model inputs before training."""
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("Training data and labels must have same length")
    if X_val.shape[0] != y_val.shape[0]:
        raise ValueError("Validation data and labels must have same length")
    if np.isnan(X_train).any() or np.isnan(y_train).any():
        raise ValueError("Training data contains NaN values")
    if np.isnan(X_val).any() or np.isnan(y_val).any():
        raise ValueError("Validation data contains NaN values")

# ---------------------------
# 8. Main Execution
# ---------------------------
if __name__ == "__main__":
    # Generate a sample dataset (replace with your actual gold futures data)
    dates = pd.date_range(start="2020-01-01", periods=300)
    # Simulate a non-linear, noisy process
    values = np.linspace(1500, 2000, 300) + np.random.normal(0, 20, 300)
    data = pd.DataFrame({'date': dates, 'value': values})
    data.set_index('date', inplace=True)
    
    # Make the data global so that the objective function can access it
    global global_data
    global_data = data.copy()
    
    # Run hyperparameter tuning with Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20, timeout=600)
    
    print("Best trial:")
    trial = study.best_trial
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    
    # Use the best hyperparameters to train the final model.
    # Note: We use the best parameters for time_step, epochs, and batch_size.
    best_params = trial.params
    final_time_step = best_params.get("time_step", 10)
    final_epochs = best_params.get("epochs", 50)
    final_batch_size = best_params.get("batch_size", 32)
    
    final_model, predictions, actual, mae, scaler = train_final_model(best_params, data, final_epochs, final_batch_size, final_time_step)
    
    print("\nSample Predictions vs. Actuals:")
    for pred, act in zip(predictions[:10], actual[:10]):
        print(f"Predicted: {pred:.2f} | Actual: {act:.2f}")
