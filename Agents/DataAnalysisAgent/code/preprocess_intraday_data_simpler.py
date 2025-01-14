import pandas as pd

# File paths (ensure these match the actual file paths)
file_paths = {
    "1min": "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\data\\intraday\\SPX_1min.csv",
    "5min": "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\data\\intraday\\SPX_5min.csv",
    "1hour": "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\data\\intraday\\SPX_1hour.csv"
}

# Process each file
for resolution, path in file_paths.items():
    try:
        # Load the CSV file
        df = pd.read_csv(path)

        # Ensure timestamp is in datetime format
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Calculate new features
        df["volatility"] = df["high"] - df["low"]
        df["average_price"] = (df["open"] + df["close"]) / 2
        df["momentum"] = df["close"] - df["open"]

        # Save the processed data
        processed_path = path.replace(".csv", "_processed.csv")
        df.to_csv(processed_path, index=False)
        print(f"{resolution} data processed and saved to {processed_path}")

    except Exception as e:
        print(f"Error processing {resolution} data: {e}")
