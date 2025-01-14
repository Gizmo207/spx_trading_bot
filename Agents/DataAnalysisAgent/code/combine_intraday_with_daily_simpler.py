import pandas as pd

# Paths to processed intraday files
intraday_files = {
    "1min": "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\data\\intraday\\SPX_1min_processed.csv",
    "5min": "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\data\\intraday\\SPX_5min_processed.csv",
    "1hour": "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\data\\intraday\\SPX_1hour_processed.csv"
}

# Path to your existing daily dataset
daily_file = "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\data\\historical_data.csv"

# Load the daily dataset
daily_data = pd.read_csv(daily_file)
daily_data["date"] = pd.to_datetime(daily_data["date"])

# Initialize an empty DataFrame for aggregated features
aggregated_features = daily_data[["date"]].copy()

# Aggregate intraday data and merge with daily data
for resolution, file_path in intraday_files.items():
    try:
        # Load intraday data
        intraday_data = pd.read_csv(file_path)
        intraday_data["timestamp"] = pd.to_datetime(intraday_data["timestamp"])
        intraday_data["date"] = intraday_data["timestamp"].dt.date

        # Aggregate features (daily summary of intraday metrics)
        aggregated = intraday_data.groupby("date").agg({
            "volatility": "mean",
            "average_price": "mean",
            "momentum": "sum"
        }).reset_index()

        # Ensure date column is in datetime format
        aggregated["date"] = pd.to_datetime(aggregated["date"])
        aggregated_features["date"] = pd.to_datetime(aggregated_features["date"])

        # Rename columns to include resolution
        aggregated.columns = ["date", f"{resolution}_volatility", f"{resolution}_average_price", f"{resolution}_momentum"]

        # Merge aggregated features into the main DataFrame
        aggregated_features = pd.merge(aggregated_features, aggregated, on="date", how="left")
        print(f"Aggregated {resolution} data successfully.")
    except Exception as e:
        print(f"Error processing {resolution} data: {e}")

# Merge aggregated intraday features with daily dataset
try:
    final_data = pd.merge(daily_data, aggregated_features, on="date", how="left")
    # Save the final dataset
    output_path = "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\data\\final_dataset_with_intraday.csv"
    final_data.to_csv(output_path, index=False)
    print(f"Final dataset saved to {output_path}")
except Exception as e:
    print(f"Error merging datasets: {e}")
