from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

from src.data_loader import DataLoader
from src.feature_eng import FeatureEngineer
from src.models import QuantileModels
from src.deep_models import DeepQuantileModel
from src.strategy import SignalGenerator

app = Flask(__name__)

@app.route("/")
def home():
    return "AI Stock Forecast API Running 🚀"


@app.route("/forecast", methods=["POST"])
def forecast():

    try:
        data = request.json
        
        TICKER = data.get("ticker", "RELIANCE.NS")
        HORIZON = int(data.get("horizon", 21))
        CONFIDENCE = float(data.get("confidence", 0.90))
        MODEL_TYPE = data.get("model_type", "Ensemble")

        SPLIT_RATIO = 0.80

        # 1️⃣ Load Data
        loader = DataLoader(TICKER)
        raw_df = loader.fetch_data()

        if raw_df is None or len(raw_df) < 200:
            return jsonify({"error": "Insufficient data"}), 400

        # 2️⃣ Feature Engineering
        fe = FeatureEngineer(raw_df)
        fe.add_technical_indicators()
        df = fe.create_targets(HORIZON)

        feature_cols = ['Close', 'VIX', 'ATR', 'BB_Width', 'Return']

        X = df[feature_cols]
        y = df['Target_Return']

        split_idx = int(len(X) * SPLIT_RATIO)

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        last_row = fe.df.iloc[[-1]][feature_cols]

        # 3️⃣ Model
        alpha_lower = (1 - CONFIDENCE) / 2

        f_low, f_high = 0, 0

        if "LightGBM" in MODEL_TYPE or "Ensemble" in MODEL_TYPE:
            qm = QuantileModels(alpha_lower, 1 - alpha_lower)
            m_low, m_high = qm.train_lgbm(X_train, y_train)
            f_low_lgb = m_low.predict(last_row)[0]
            f_high_lgb = m_high.predict(last_row)[0]

        if "LSTM" in MODEL_TYPE or "Ensemble" in MODEL_TYPE:
            dl = DeepQuantileModel(input_shape=(1, len(feature_cols)))
            dl.train(X_train, y_train, epochs=20)
            f_low_lstm, f_high_lstm = dl.predict(last_row)
            f_low_lstm = f_low_lstm[0]
            f_high_lstm = f_high_lstm[0]

        if "Ensemble" in MODEL_TYPE:
            f_low = (f_low_lgb + f_low_lstm) / 2
            f_high = (f_high_lgb + f_high_lstm) / 2
        elif "LightGBM" in MODEL_TYPE:
            f_low, f_high = f_low_lgb, f_high_lgb
        else:
            f_low, f_high = f_low_lstm, f_high_lstm

        # 4️⃣ Convert to Price
        latest_price = raw_df['Close'].iloc[-1]

        p_low = latest_price * (1 + f_low)
        p_high = latest_price * (1 + f_high)

        return jsonify({
            "ticker": TICKER,
            "current_price": float(latest_price),
            "bearish_limit": float(p_low),
            "bullish_limit": float(p_high),
            "confidence": CONFIDENCE,
            "horizon_days": HORIZON
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
