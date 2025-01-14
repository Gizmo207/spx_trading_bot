from google.cloud import firestore
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Initialize Firestore
db = firestore.Client()

def fetch_historical_data():
    """
    Fetch historical data from Firestore and load it into a DataFrame.
    """
    # Use the correct collection name
    collection_ref = db.collection("historical_data")
    docs = collection_ref.stream()

    data = []
    for doc in docs:
        entry = doc.to_dict()
        data.append(entry)
    
    # Convert the list of data into a DataFrame
    df = pd.DataFrame(data)
    return df

def preprocess_data(df):
    """
    Preprocess the historical data:
    1. Fill missing values using backfill and forward fill.
    2. Normalize numerical features.
    """
    # Step 1: Handle missing values
    df = df.bfill()  # Backfill
    df = df.ffill()  # Forward fill

    # Step 2: Normalize features (scaling to 0–1 range)
    scaler = MinMaxScaler()

    # Select numerical columns for normalization
    numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df

def main():
    """
    Main function to fetch, preprocess, and display the data.
    """
    # Fetch data from Firestore
    historical_data = fetch_historical_data()

    # Preprocess the data
    preprocessed_data = preprocess_data(historical_data)

    # Display the preprocessed data
    print("Preprocessed data:")
    print(preprocessed_data.head())

if __name__ == "__main__":
    main()
