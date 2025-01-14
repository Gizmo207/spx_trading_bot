from google.cloud import firestore  # Firestore library
import pandas as pd
import joblib
import logging
from datetime import datetime

# Configure logging
log_file = "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\logs\\prediction.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def load_model():
    """
    Load the trained Random Forest model from the models folder.
    """
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

def load_all_data():
    """
    Load all historical data from the preprocessed CSV file.
    """
    try:
        file_path = "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\data\\preprocessed_data.csv"
        df = pd.read_csv(file_path)
        logging.info("All historical data loaded successfully.")
        print("All historical data loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        print(f"Error loading data: {e}")
        raise

def preprocess_data(df):
    """
    Preprocess the data for prediction by selecting relevant features.
    """
    try:
        features = ["open", "high", "low", "close", "volume", "rsi", "macd", "bollinger_upper", "bollinger_lower"]
        logging.info("Data preprocessed for prediction.")
        return df[features]
    except KeyError as e:
        logging.error(f"Missing required column: {e}")
        print(f"Missing required column: {e}")
        raise
    except Exception as e:
        logging.error(f"Error during data preprocessing: {e}")
        print(f"Error during data preprocessing: {e}")
        raise

def predict_sentiment(model, data):
    """
    Use the trained model to predict bullish/bearish sentiment for all data.
    """
    try:
        predictions = model.predict(data)
        sentiments = ["Bullish" if p == 1 else "Bearish" for p in predictions]
        logging.info("Predictions generated successfully.")
        print("Predictions generated successfully.")
        return sentiments
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        print(f"Error during prediction: {e}")
        raise

def log_predictions_to_firestore(predictions, data):
    """
    Log all predictions to Firestore in the daily_calls collection.
    """
    try:
        # Initialize Firestore client
        db = firestore.Client()
        collection_ref = db.collection("daily_calls")

        for i, prediction in enumerate(predictions):
            # Prepare document data
            document_data = {
                "date": data.iloc[i].get("date", "unknown"),
                "prediction": prediction,
                "reason": {
                    "rsi": data.iloc[i].get("rsi", None),
                    "macd": data.iloc[i].get("macd_signal", None),
                    "bollinger_upper": data.iloc[i].get("bollinger_upper", None),
                    "bollinger_lower": data.iloc[i].get("bollinger_lower", None)
                }
            }

            # Debugging: Print the data being sent
            print(f"Logging document to Firestore: {document_data}")
            logging.info(f"Logging document to Firestore: {document_data}")

            # Send data to Firestore
            doc_ref = collection_ref.document()
            doc_ref.set(document_data)

        logging.info(f"{len(predictions)} predictions logged to Firestore.")
        print(f"{len(predictions)} predictions logged to Firestore.")
    except Exception as e:
        logging.error(f"Error logging predictions to Firestore: {e}")
        print(f"Error logging predictions to Firestore: {e}")
        raise

def main():
    """
    Main function to predict sentiment for all data and log it to Firestore.
    """
    try:
        # Load the trained model
        model = load_model()

        # Load all historical data
        all_data = load_all_data()

        # Preprocess the data
        preprocessed_data = preprocess_data(all_data)

        # Predict sentiments
        sentiments = predict_sentiment(model, preprocessed_data)

        # Log predictions to Firestore
        log_predictions_to_firestore(sentiments, all_data)

        # Indicate script has completed successfully
        print("All predictions logged to Firestore successfully!")
        logging.info("All predictions logged to Firestore successfully!")
    except Exception as e:
        logging.error(f"Unhandled error in main: {e}")
        print(f"Unhandled error: {e}")

if __name__ == "__main__":
    main()
