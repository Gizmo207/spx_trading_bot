import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime
import yfinance as yf
import firebase_admin
from firebase_admin import credentials, firestore
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration
CSV_PATH = r"C:\Users\inves\OneDrive\Documents\Desktop\TradingBotProject\Agents\DataAnalysisAgent\data\final_dataset_with_intraday.csv"
FIREBASE_CRED_PATH = r"C:\Users\inves\OneDrive\Documents\Desktop\TradingBotProject\Shared\configs\firebase-key.json"
MODEL_PATH = r"C:\Users\inves\OneDrive\Documents\Desktop\TradingBotProject\Agents\DataAnalysisAgent\models\lstm_model.h5"
NUM_FEATURES = 30  # Update this based on your model's expected input size
DAILY_DATA_COLLECTION = "daily_data"
DAILY_CALL_COLLECTION = "daily_call"

# Initialize Firebase
try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    logging.info("Firebase initialized successfully.")
except Exception as e:
    logging.error(f"Error initializing Firebase: {e}")

# Fetch latest data from Yahoo Finance
def fetch_latest_data():
    logging.info("Fetching the latest data from Yahoo Finance...")
    try:
        data = yf.download("^GSPC", period="1d", interval="1d")
        latest_data = data.iloc[-1]
        latest_date = data.index[-1]
        logging.info(f"Latest data fetched: {latest_data.to_dict()}")
        return latest_date, latest_data
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None, None

# Save data to Firestore
def save_to_firestore(collection, document_id, data):
    try:
        db.collection(collection).document(document_id).set(data)
        logging.info(f"Data saved to Firestore collection '{collection}' with ID '{document_id}'.")
    except Exception as e:
        logging.error(f"Error saving data to Firestore: {e}")

# Append data to CSV
def append_to_csv(latest_date, latest_data):
    try:
        new_row = {
            "date": latest_date.strftime("%Y-%m-%d"),
            "close": float(latest_data["Close"]),
            "high": float(latest_data["High"]),
            "low": float(latest_data["Low"]),
            "open": float(latest_data["Open"]),
            "volume": int(latest_data["Volume"]),
        }
        for i in range(1, NUM_FEATURES - len(new_row) + 1):
            new_row[f"feature_{i}"] = 0

        df = pd.DataFrame([new_row])
        df.to_csv(CSV_PATH, mode="a", header=False, index=False)
        logging.info("Data appended to CSV successfully.")
    except Exception as e:
        logging.error(f"Error appending data to CSV: {e}")

# Make prediction
def make_prediction():
    logging.info("Loading dataset for prediction...")
    try:
        data = pd.read_csv(CSV_PATH)
        columns_to_drop = ["prediction", "date", "Ticker"]
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns], errors="ignore")
        data = data.apply(pd.to_numeric, errors="coerce").fillna(0)

        if data.shape[1] != NUM_FEATURES:
            raise ValueError(f"Expected {NUM_FEATURES} features, but got {data.shape[1]}.")

        logging.info("Dataset loaded and validated successfully.")
        model = load_model(MODEL_PATH)
        logging.info("Model loaded successfully.")

        input_data = np.expand_dims(data.values[-1:], axis=0)
        prediction = model.predict(input_data)
        predicted_label = "Bullish" if prediction[0][0] >= 0.5 else "Bearish"
        confidence = prediction[0][0] if prediction[0][0] >= 0.5 else 1 - prediction[0][0]

        logging.info(f"Prediction: {predicted_label} (Confidence: {confidence * 100:.2f}%)")
        return predicted_label, confidence
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        return None, None

# Main script
if __name__ == "__main__":
    latest_date, latest_data = fetch_latest_data()
    if latest_date is not None and latest_data is not None:
        save_to_firestore(
            DAILY_DATA_COLLECTION,
            latest_date.strftime("%Y-%m-%d"),
            {
                "close": float(latest_data["Close"]),
                "high": float(latest_data["High"]),
                "low": float(latest_data["Low"]),
                "open": float(latest_data["Open"]),
                "volume": int(latest_data["Volume"]),
                "timestamp": datetime.now().isoformat(),
            },
        )
        append_to_csv(latest_date, latest_data)
        predicted_label, confidence = make_prediction()
        if predicted_label is not None:
            save_to_firestore(
                DAILY_CALL_COLLECTION,
                latest_date.strftime("%Y-%m-%d"),
                {
                    "date": latest_date.strftime("%Y-%m-%d"),
                    "prediction": predicted_label,
                    "confidence": confidence,
                    "timestamp": datetime.now().isoformat(),
                },
            )
