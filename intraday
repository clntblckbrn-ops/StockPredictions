!pip install gradio yfinance pandas scikit-learn numpy --quiet

import gradio as gr
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import warnings

# Ignore common warnings
warnings.filterwarnings('ignore')

def process_single_ticker(ticker):
    """
    Core logic for processing one ticker. Returns a formatted string.
    """
    try:
        # 1. --- Get Data ---
        data = yf.download(ticker, period="10y", progress=False)
        if data.empty:
            return f"--- {ticker.upper()} ---\nError: No data found."

        # 2. --- Feature Engineering (Simplified & Reliable) ---
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_30'] = data['Close'].rolling(window=30).mean()
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        data['Lag_Return_1'] = data['Close'].pct_change(1)
        data['Lag_Return_5'] = data['Close'].pct_change(5)
        
        # 3. --- Define Target and Features ---
        data['Target'] = np.where(data['Close'].shift(-1) > data['Open'].shift(-1), 1, 0)
        data.dropna(inplace=True)

        if len(data) < 50:
            return f"--- {ticker.upper()} ---\nError: Not enough historical data."

        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                    'SMA_10', 'SMA_30', 'RSI', 'Lag_Return_1', 'Lag_Return_5']
        X = data[features]
        y = data['Target']
        
        # 4. --- Train Model & Predict ---
        model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=42).fit(X, y)
        last_row = X.iloc[[-1]]
        prediction = model.predict(last_row)[0]
        probability = model.predict_proba(last_row)[0]

        # 5. --- Return Result String ---
        return (f"--- {ticker.upper()} ---\n"
                f"Predicted Intraday Movement: {'HIGHER' if prediction == 1 else 'LOWER'}\n"
                f"Model Confidence: {probability[prediction]:.2%}")

    except Exception as e:
        return f"--- {ticker.upper()} ---\nAn unexpected error occurred: {e}"

def predict_for_gradio(tickers_string):
    """
    Main Gradio function that processes the input string.
    """
    if not tickers_string:
        return "Please enter at least one ticker symbol."

    tickers = [ticker.strip().upper() for ticker in tickers_string.split(',')]
    if len(tickers) > 10:
        tickers = tickers[:10]
        limit_message = "Processing the first 10 tickers.\n\n"
    else:
        limit_message = ""

    all_results = [process_single_ticker(ticker) for ticker in tickers]
    separator = "\n\n" + "="*40 + "\n\n"
    return limit_message + separator.join(all_results)

# --- Create and Launch the Gradio App ---
iface = gr.Interface(
    fn=predict_for_gradio,
    inputs=gr.Textbox(
        label="Enter up to 10 Stock Tickers (comma-separated)",
        placeholder="e.g., SPY, QQQ, TSLA"
    ),
    outputs=gr.Textbox(label="Prediction Results", lines=25),
    title="Reliable Intraday Stock Predictor 📈",
    description="This simplified model predicts if a stock will close higher or lower than its open for the upcoming session. Enter tickers to get your forecast."
)

iface.launch()
