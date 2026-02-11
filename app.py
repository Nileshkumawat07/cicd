from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.data_loader import DataLoader
from src.feature_eng import FeatureEngineer
from src.models import QuantileModels
from src.deep_models import DeepQuantileModel
from src.strategy import SignalGenerator

app = Flask(__name__)

# ===============================
# Utility: Create Plotly Chart
# ===============================

def create_chart(dates, actual, lower, upper):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=actual,
        mode='lines',
        name='Actual Price',
        line=dict(color='#2563eb', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=upper,
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=lower,
        fill='tonexty',
        fillcolor='rgba(16,185,129,0.2)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Confidence Band'
    ))

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, t=20, b=0),
        height=320
    )

    graph_html = fig.to_html(
        full_html=False,
        include_plotlyjs='cdn',
        config={
            "displayModeBar": False,
            "scrollZoom": False
        }
    )

    return graph_html


# ===============================
# Main Route (Dashboard)
# ===============================

@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    error = None
    graph_html = None

    if request.method == "POST":
        try:
            ticker = request.form["ticker"].upper()
            horizon = int(request.form["horizon"])
            confidence = float(request.form["confidence"])
            model_type = request.form["model_type"]

            split_ratio = 0.80

            # =========================
            # 1. Load Data
            # =========================
            loader = DataLoader(ticker)
            raw_df = loader.fetch_data()

            if raw_df is None or len(raw_df) < 200:
                error = "Insufficient historical data."
                return render_template("index.html", result=None, error=error)

            # =========================
            # 2. Feature Engineering
            # =========================
            fe = FeatureEngineer(raw_df)
            fe.add_technical_indicators()
            df = fe.create_targets(horizon)

            feature_cols = ['Close', 'VIX', 'ATR', 'BB_Width', 'Return']

            X = df[feature_cols]
            y = df['Target_Return']

            split_idx = int(len(X) * split_ratio)

            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]

            last_row = fe.df.iloc[[-1]][feature_cols]

            alpha_lower = (1 - confidence) / 2

            # =========================
            # 3. Model Logic
            # =========================
            f_low = f_high = 0
            pred_low = pred_high = 0

            # ----- LightGBM -----
            if model_type in ["LightGBM", "Ensemble"]:
                qm = QuantileModels(alpha_lower, 1 - alpha_lower)
                m_low, m_high = qm.train_lgbm(X_train, y_train)

                f_low_lgb = m_low.predict(last_row)[0]
                f_high_lgb = m_high.predict(last_row)[0]

                pred_low_lgb = m_low.predict(X_test)
                pred_high_lgb = m_high.predict(X_test)

            # ----- LSTM -----
            if model_type in ["LSTM", "Ensemble"]:
                dl = DeepQuantileModel(input_shape=(1, len(feature_cols)))
                dl.train(X_train, y_train, epochs=10)

                f_low_lstm, f_high_lstm = dl.predict(last_row)
                f_low_lstm = f_low_lstm[0]
                f_high_lstm = f_high_lstm[0]

                pred_low_lstm, pred_high_lstm = dl.predict(X_test)

            # ----- Ensemble -----
            if model_type == "Ensemble":
                f_low = (f_low_lgb + f_low_lstm) / 2
                f_high = (f_high_lgb + f_high_lstm) / 2
                pred_low = (pred_low_lgb + pred_low_lstm) / 2
                pred_high = (pred_high_lgb + pred_high_lstm) / 2

            elif model_type == "LightGBM":
                f_low, f_high = f_low_lgb, f_high_lgb
                pred_low, pred_high = pred_low_lgb, pred_high_lgb

            else:
                f_low, f_high = f_low_lstm, f_high_lstm
                pred_low, pred_high = pred_low_lstm, pred_high_lstm

            # =========================
            # 4. Final Forecast Values
            # =========================
            latest_price = raw_df['Close'].iloc[-1]

            p_low = latest_price * (1 + f_low)
            p_high = latest_price * (1 + f_high)

            # =========================
            # 5. Backtest
            # =========================
            current_prices = df['Close'].iloc[split_idx:].values

            pl_price = current_prices * (1 + pred_low)
            ph_price = current_prices * (1 + pred_high)
            y_price = current_prices * (1 + y_test.values)

            strat = SignalGenerator()
            _, signals, pnl = strat.run_mean_reversion(
                df.index[split_idx:], y_price, pl_price, ph_price
            )

            picp = ((y_price >= pl_price) & (y_price <= ph_price)).mean()

            # =========================
            # 6. Chart
            # =========================
            graph_html = create_chart(
                df.index[split_idx:], 
                y_price, 
                pl_price, 
                ph_price
            )

            # =========================
            # 7. Result Object
            # =========================
            result = {
                "ticker": ticker,
                "model": model_type,
                "price": round(latest_price, 2),
                "bearish": round(p_low, 2),
                "bullish": round(p_high, 2),
                "coverage": round(picp * 100, 2),
                "pnl": round(pnl, 2)
            }

        except Exception as e:
            error = str(e)

    return render_template("index.html", result=result, error=error, graph_html=graph_html)


# ===============================
# API Route
# ===============================

@app.route("/api/forecast", methods=["POST"])
def forecast_api():
    try:
        data = request.json
        ticker = data["ticker"].upper()
        horizon = int(data["horizon"])

        loader = DataLoader(ticker)
        raw_df = loader.fetch_data()

        latest_price = raw_df['Close'].iloc[-1]

        return jsonify({
            "ticker": ticker,
            "current_price": float(latest_price),
            "message": "API working"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===============================
# Run Server
# ===============================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
