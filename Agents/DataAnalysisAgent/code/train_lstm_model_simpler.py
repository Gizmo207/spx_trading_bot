import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
file_path = "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\data\\cleaned_dataset.csv"
print("Loading dataset...")
data = pd.read_csv(file_path)

# Define features and target
print("Preparing features and target...")
features = data.drop(columns=["date", "prediction"], errors="ignore")
target = data["prediction"].map({"Bullish": 1, "Bearish": 0})  # Convert labels to numeric

# Ensure all columns are numeric and handle missing values
print("Ensuring all columns are numeric...")
features = features.apply(pd.to_numeric, errors="coerce")
features.fillna(features.mean(), inplace=True)

# Check for invalid values
if features.isnull().any().any():
    print("Error: Features contain nan or invalid values!")
    exit()
if target.isnull().any():
    print("Error: Target contains nan or invalid values!")
    exit()

# Normalize features
print("Normalizing features...")
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Reshape for LSTM input (samples, timesteps, features)
X = features_scaled.reshape((features_scaled.shape[0], 1, features_scaled.shape[1]))

# Split data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)

# Build LSTM model
print("Building the LSTM model...")
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

# Compile the model
print("Compiling the model...")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# Train the model
print("Training the model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
print("Evaluating the model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Save the model
model_path = "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\models\\lstm_model.h5"
model.save(model_path)
print(f"Model saved to {model_path}")
