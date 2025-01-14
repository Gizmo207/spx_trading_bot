from google.cloud import firestore
import pandas as pd
import joblib
import logging
import os

# Configure logging
log_file = "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\logs\\prediction.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def load_model():
    """Load the trained Random Forest model."""
    try:
        model_path = "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\models\\random_forest_model.pkl"
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
        print("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}")
        raise

def load_csv_data():
    """Load all historical data from the preprocessed CSV file."""
    try:
        file_path = "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\data\\final_dataset_with_intraday.csv"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        df = pd.read_csv(file_path)
        logging.info("CSV data loaded successfully.")
        print("CSV data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV data: {e}")
        print(f"Error loading CSV data: {e}")
        raise

def predict_next_day(model, df):
    """
    Predict the sentiment for the next trading day using all historical data.
    Also return confidence and richer explanations.
    """
    try:
        # Extract features
        features = ["open", "high", "low", "close", "volume", "('Close', '^GSPC')", "('High', '^GSPC')", "('Low', '^GSPC')", "('Open', '^GSPC')", "('Volume', '^GSPC')"]
        X = df[features]
        
        # Predict for the next day
        prediction_proba = model.predict_proba(X)[-1]  # Probabilities for the last row
        prediction = model.predict(X)[-1]  # Final prediction
        sentiment = "Bullish" if prediction == 1 else "Bearish"
        confidence = max(prediction_proba) * 100  # Convert to percentage
        
        # Extract metrics for the latest day
        latest_metrics = df.iloc[-1].to_dict()

        # Create a richer explanation
        decision_reason = (
            f"RSI={latest_metrics.get('rsi')}, MACD={latest_metrics.get('macd')}, "
            f"Bollinger Upper={latest_metrics.get('bollinger_upper')}, "
            f"Bollinger Lower={latest_metrics.get('bollinger_lower')}, "
            f"Volume={latest_metrics.get('volume')}. "
            "These indicators suggest the market is "
            + ("overbought and due for a pullback." if sentiment == "Bearish" else "bullish with positive momentum.")
        )

        logging.info(f"Prediction for next trading day: {sentiment}, Confidence: {confidence:.2f}%")
        print(f"Prediction for next trading day: {sentiment}, Confidence: {confidence:.2f}%")

        return sentiment, confidence, latest_metrics, decision_reason
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        print(f"Error during prediction: {e}")
        raise

def log_to_firestore(sentiment, confidence, metrics, decision_reason):
    """Log the final prediction and details to Firestore."""
    try:
        db = firestore.Client()
        collection_ref = db.collection("daily_call")
        collection_ref.add({
            "date": firestore.SERVER_TIMESTAMP,
            "prediction": sentiment,
            "confidence": f"{confidence:.2f}%",
            "metrics": {
                "rsi": metrics.get("rsi"),
                "macd": metrics.get("macd"),
                "bollinger_upper": metrics.get("bollinger_upper"),
                "bollinger_lower": metrics.get("bollinger_lower"),
                "volume": metrics.get("volume"),
                "close": metrics.get("close")
            },
            "decision_reason": decision_reason,
            "model_version": "1.0.0"  # Example model version
        })
        logging.info(f"Prediction logged to Firestore: {sentiment}")
        print(f"Prediction logged to Firestore: {sentiment}")
    except Exception as e:
        logging.error(f"Error logging prediction to Firestore: {e}")
        print(f"Error logging prediction to Firestore: {e}")
        raise

def main():
    """Main function to load data, make a prediction, and log the result."""
    try:
        # Load the trained model
        model = load_model()

        # Load the CSV data
        df = load_csv_data()

        # Predict the sentiment for the next trading day
        sentiment, confidence, metrics, decision_reason = predict_next_day(model, df)

        # Log the prediction to Firestore
        log_to_firestore(sentiment, confidence, metrics, decision_reason)

        print("Prediction process completed successfully.")
        logging.info("Prediction process completed successfully.")
    except Exception as e:
        logging.error(f"Unhandled error: {e}")
        print(f"Unhandled error: {e}")

if __name__ == "__main__":
    main()
import os

def load_csv_data():
    """Load all historical data from the preprocessed CSV file."""
    try:
        file_path = "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\data\\final_dataset_with_intraday.csv"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        df = pd.read_csv(file_path, error_bad_lines=False)
        logging.info("CSV data loaded successfully.")
        print("CSV data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV data: {e}")
        print(f"Error loading CSV data: {e}")
        raise        import os
        
        def load_csv_data():
            """Load all historical data from the preprocessed CSV file."""
            try:
                file_path = "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\data\\final_dataset_with_intraday.csv"
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")
                df = pd.read_csv(file_path, error_bad_lines=False)
                logging.info("CSV data loaded successfully.")
                print("CSV data loaded successfully.")
                return df
            except Exception as e:
                logging.error(f"Error loading CSV data: {e}")
                print(f"Error loading CSV data: {e}")
                raise