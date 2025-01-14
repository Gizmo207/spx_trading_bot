import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib  # For saving the trained model

def load_data():
    """
    Load the preprocessed data from the CSV file and add the bullish_bearish column.
    """
    file_path = "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\data\\preprocessed_data.csv"
    df = pd.read_csv(file_path)

    # Add the target column: 1 for bullish, 0 for bearish
    df["bullish_bearish"] = (df["close"] > df["open"]).astype(int)

    return df

def train_model(df):
    """
    Train a Random Forest classifier to predict bullish or bearish days.
    """
    # Step 1: Define features (X) and target (y)
    features = ["open", "high", "low", "close", "volume", "rsi", "macd", "bollinger_upper", "bollinger_lower"]
    target = "bullish_bearish"

    X = df[features]
    y = df[target]

    # Step 2: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 4: Evaluate the model
    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return model

def save_model(model):
    """
    Save the trained model to the models folder.
    """
    output_path = "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\models\\random_forest_model.pkl"
    joblib.dump(model, output_path)
    print(f"Model saved to {output_path}")

def main():
    """
    Main function to load data, train the model, and save it.
    """
    # Load the data
    data = load_data()

    # Train the model
    model = train_model(data)

    # Save the trained model
    save_model(model)

if __name__ == "__main__":
    main()
