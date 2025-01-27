import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data.set_index('timestamp', inplace=True)
    return data

# ARIMA model for linear patterns
def fit_arima(data, order=(1, 1, 1)):
    model = ARIMA(data, order=order)
    arima_fit = model.fit()
    return arima_fit

# Prepare data for neural network
def prepare_nn_data(residuals, original_data, future_steps=15):
    residuals = residuals[~np.isnan(residuals)]  # Drop NaN residuals
    X, y = [], []
    for i in range(len(residuals) - future_steps):
        X.append(residuals[i:i + future_steps])
        y.append(original_data[i + future_steps])
    return np.array(X), np.array(y)

# Build LSTM neural network
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape, return_sequences=False))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Predict single value (price)
    model.compile(optimizer='adam', loss='mse')
    return model

# Combine ARIMA and LSTM predictions
def predict_hybrid(arima_fit, lstm_model, data, scaler, future_steps=15):
    arima_forecast = arima_fit.forecast(steps=future_steps)[-1]  # ARIMA prediction
    nn_input = data[-future_steps:]  # Last residuals for NN input
    nn_input = scaler.transform(nn_input.reshape(-1, 1)).reshape(1, -1, 1)
    nn_forecast = lstm_model.predict(nn_input)
    return arima_forecast + scaler.inverse_transform(nn_forecast).flatten()[0]

# Main workflow
def main(file_path):
    # Step 1: Load data
    data = load_data(file_path)
    close_prices = data['close']

    # Step 2: Fit ARIMA
    arima_fit = fit_arima(close_prices)

    # Step 3: Extract ARIMA residuals
    residuals = arima_fit.resid

    # Step 4: Scale residuals for NN
    scaler = MinMaxScaler()
    residuals_scaled = scaler.fit_transform(residuals.values.reshape(-1, 1))

    # Step 5: Prepare data for NN
    X, y = prepare_nn_data(residuals_scaled, close_prices.values)

    # Reshape X for LSTM input (samples, timesteps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Step 6: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 7: Build and train LSTM model
    lstm_model = build_lstm_model(X_train.shape[1:])
    lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Step 8: Predict using hybrid model
    hybrid_prediction = predict_hybrid(arima_fit, lstm_model, residuals.values, scaler)
    print(f"Hybrid Prediction for 15 minutes later: {hybrid_prediction}")

# Run the script
if __name__ == "__main__":
    file_path = r"C:\Aviral\GITHUB\Price-Prediction\backtesting\data\nifty2015-2025.csv"  # Replace with your file path
    main(file_path)
