import pandas as pd

# Load the dataset
file_path = "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\data\\cleaned_dataset.csv"
data = pd.read_csv(file_path)

# Fill missing values in 'prediction' column
print("Populating 'prediction' column with rule-based logic...")

# Example rule-based logic: Assign 'Bullish' if close > open, otherwise 'Bearish'
data["prediction"] = data.apply(lambda row: "Bullish" if row["close"] > row["open"] else "Bearish", axis=1)

# Check for remaining missing values
if data["prediction"].isnull().any():
    print("Error: 'prediction' column still contains missing values!")
else:
    print("Successfully populated 'prediction' column.")

# Save the updated dataset
output_path = "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\data\\cleaned_dataset.csv"
data.to_csv(output_path, index=False)
print(f"Updated dataset saved to {output_path}")
