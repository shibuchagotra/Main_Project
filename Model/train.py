from prophet import Prophet
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import joblib
import re
from prophet.serialize import model_to_json, model_from_json
import os
import matplotlib.pyplot as plt


class Airpollution:
    def __init__(self, target, feature):
        self.target = target
        self.feature = feature
        self.seq_length = None
        self.features = None
        self.last_result = None

        # sanitize pollutant name for folder & filenames
        self.safe_target = re.sub(r'[^A-Za-z0-9_]+', '_', target)
        self.base_dir = f"{self.safe_target}"
        os.makedirs(self.base_dir, exist_ok=True)

        # paths inside pollutant folder
        self.prophet_file = os.path.join(self.base_dir, f"prophet_model_{self.safe_target}.json")
        self.lstm_file = os.path.join(self.base_dir, f"lstm_model_{self.safe_target}.h5")
        self.scaler_file = os.path.join(self.base_dir, f"scaler_{self.safe_target}.pkl")

    # create sequences for LSTM
    def create_sequences(self, data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length, :])
            y.append(data[i + seq_length, 0])
        return np.array(X), np.array(y)

    # Train Prophet + LSTM hybrid model
    def train(self, train_df, seq_length, epoch, batch):
        self.seq_length = seq_length
        self.features = ['residual'] + self.feature

        # --- Prophet ---
        prophet_df = train_df[['Timestamp', self.target]].rename(columns={'Timestamp': 'ds', self.target: 'y'}).dropna()
        prophet_model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
        prophet_model.fit(prophet_df)

        # Save Prophet model
        with open(self.prophet_file, "w") as f:
            f.write(model_to_json(prophet_model))

        # Forecast on training data
        forecast_train = prophet_model.predict(prophet_df)
        train_df = train_df.merge(forecast_train[['ds', 'yhat']], left_on='Timestamp', right_on='ds', how='left')
        train_df.rename(columns={'yhat': 'prophet_yhat'}, inplace=True)
        train_df['residual'] = train_df[self.target] - train_df['prophet_yhat']
        train_df.drop(columns=['ds'], inplace=True)

        # Fill missing residuals
        train_df['residual'] = (
            train_df['residual']
            .replace([np.inf, -np.inf], np.nan)
            .interpolate()
            .fillna(method='bfill')
            .fillna(method='ffill')
        )
        train_df[self.features] = train_df[self.features].interpolate().fillna(method='bfill').fillna(method='ffill')

        # --- Save Prophet plots ---
        fig1 = prophet_model.plot(forecast_train)
        plt.title(f"{self.target} Forecast (Training Data)")
        plt.xlabel("Date")
        plt.ylabel(self.target)
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, f"prophet_forecast_{self.safe_target}.png"))
        plt.close(fig1)

        fig2 = prophet_model.plot_components(forecast_train)
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, f"prophet_components_{self.safe_target}.png"))
        plt.close(fig2)

        # --- Scale features ---
        data = train_df[self.features].values
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_scaled = scaler.fit_transform(data)
        joblib.dump(scaler, self.scaler_file)

        # --- LSTM sequences ---
        X, y = self.create_sequences(data_scaled, self.seq_length)
        model_lstm = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_length, X.shape[2])),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])

        model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse')


        # Train LSTM
        model_lstm.fit(X, y, epochs=epoch, batch_size=batch, verbose=1)

        # Save LSTM model
        model_lstm.save(self.lstm_file)

        print(f"✅ Training completed! Models & plots saved in '{self.base_dir}/'")

    # Test
    def test(self, test_df):
        # Load models
        if not os.path.exists(self.lstm_file) or not os.path.exists(self.scaler_file) or not os.path.exists(self.prophet_file):
            raise FileNotFoundError("Models or scaler not found. Run train() first.")

        lstm_model = load_model(self.lstm_file, compile=False)
        scaler = joblib.load(self.scaler_file)
        with open(self.prophet_file, "r") as f:
            prophet_model = model_from_json(f.read())

        # Ensure datetime
        test_df = test_df.sort_values('Timestamp')
        test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])

        # Prophet predictions
        prophet_input = test_df[['Timestamp', self.target]].rename(columns={'Timestamp': 'ds', self.target: 'y'})
        forecast = prophet_model.predict(prophet_input)
        test_df['prophet_yhat'] = forecast['yhat'].values
        test_df['prophet_yhat'] = test_df['prophet_yhat'].fillna(method='ffill').fillna(method='bfill')

        # Compute residuals
        test_df['residual'] = test_df[self.target] - test_df['prophet_yhat']

        # Features
        test_data = test_df[self.features].values
        test_data_scaled = scaler.transform(test_data)

        # Sequences
        X_test, y_test = self.create_sequences(test_data_scaled, self.seq_length)
        print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)

        # LSTM predictions
        lstm_preds_scaled = lstm_model.predict(X_test, verbose=1)
        num_features = X_test.shape[2]
        lstm_residuals = scaler.inverse_transform(
            np.concatenate([lstm_preds_scaled, np.zeros((len(lstm_preds_scaled), num_features - 1))], axis=1)
        )[:, 0]

        # Align with test dataframe
        test_df_hybrid = test_df.iloc[self.seq_length:].copy().reset_index(drop=True)
        min_len = min(len(lstm_residuals), len(test_df_hybrid))
        test_df_hybrid = test_df_hybrid.iloc[:min_len].copy()
        lstm_residuals = lstm_residuals[:min_len]

        # Hybrid forecast
        test_df_hybrid['lstm_residual'] = lstm_residuals
        test_df_hybrid['hybrid_yhat'] = test_df_hybrid['prophet_yhat'] + test_df_hybrid['lstm_residual']

        # Save hybrid plot
        plt.figure(figsize=(10, 6))
        plt.plot(test_df_hybrid['Timestamp'], test_df_hybrid[self.target], label='Actual', color='blue')
        plt.plot(test_df_hybrid['Timestamp'], test_df_hybrid['prophet_yhat'], label='Prophet Forecast', color='orange')
        plt.plot(test_df_hybrid['Timestamp'], test_df_hybrid['hybrid_yhat'], label='Hybrid Forecast', color='green')
        plt.title(f"Prophet vs Hybrid Forecast ({self.target})")
        plt.xlabel("Date")
        plt.ylabel(self.target)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, f"hybrid_forecast_{self.safe_target}.png"))
        plt.close()

        self.last_result = test_df_hybrid
        print(test_df_hybrid[['Timestamp', self.target, 'prophet_yhat', 'hybrid_yhat']].head())
        return test_df_hybrid

    # Metrics
    def calculate_metrics(self, result_df=None):
        if result_df is None:
            if self.last_result is None:
                raise ValueError("No test results available. Run test() first.")
            result_df = self.last_result

        y_true = result_df[self.target].values
        y_hybrid = result_df['hybrid_yhat'].values

        mae = mean_absolute_error(y_true, y_hybrid)
        rmse = np.sqrt(mean_squared_error(y_true, y_hybrid))
        mape = np.mean(np.abs((y_true - y_hybrid) / y_true)) * 100

        # Print metrics
        print("Performance Metrics:")
        print(f"{'Model':<10} {'MAE':>10} {'RMSE':>10} {'MAPE (%)':>10}")
        print(f"{'Hybrid':<10} {mae:10.2f} {rmse:10.2f} {mape:10.2f}")

        # Save metrics to txt file in pollutant folder
        metrics_path = os.path.join(self.base_dir, f"metrics_{self.safe_target}.txt")
        with open(metrics_path, "w") as f:
            f.write("Performance Metrics:\n")
            f.write(f"{'Model':<10} {'MAE':>10} {'RMSE':>10} {'MAPE (%)':>10}\n")
            f.write(f"{'Hybrid':<10} {mae:10.2f} {rmse:10.2f} {mape:10.2f}\n")

        print(f"✅ Metrics saved at: {metrics_path}")


    def run_all(self, test_df):
        self.test(test_df)
        self.calculate_metrics()
if __name__ == "__main__":
    import pandas as pd
    df=pd.read_csv('D:\Coding_folder\Main_Project\Chatbot\Langgraph_Chatbot\Backend\Data\AQ_met_data.csv')
# Filter the specific row
    filtered_df = df[
    (df["State"] == "Delhi") &
    (df["City"] == "Delhi") &
    (df["Station"] == "IHBAS, Dilshad Garden, Delhi - CPCB")
]
    filtered_df['Timestamp'] = pd.to_datetime(filtered_df['Timestamp'])
    train_df = filtered_df[filtered_df['Timestamp'].dt.year <= 2023]
    test_df  = filtered_df[filtered_df['Timestamp'].dt.year == 2024]
    feature = ['RH (%)','WS (m/s)','WD (deg)']
    target='PM10 (µg/m³)'
    trainer=Airpollution(target,feature)
    seq_length=24
    epochs=80# 80 epchos for ozone
    batch_size=30
    trainer.train(train_df,seq_length,epochs,batch_size)
    trainer.run_all(test_df)