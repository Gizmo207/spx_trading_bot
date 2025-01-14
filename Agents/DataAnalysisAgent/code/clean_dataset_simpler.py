import pandas as pd

# Load the dataset
file_path = "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\data\\final_dataset_with_intraday.csv"
data = pd.read_csv(file_path)

# Drop irrelevant columns
data.drop(columns=["Unnamed: 40"], inplace=True, errors="ignore")

# Convert all columns to numeric where possible
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Handle missing data
# Drop columns with too many missing values (threshold: 90% missing)
threshold = len(data) * 0.9
data.dropna(axis=1, thresh=threshold, inplace=True)

# Fill remaining missing values with column means
data.fillna(data.mean(), inplace=True)

# Save cleaned dataset
cleaned_file_path = "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\data\\cleaned_dataset.csv"
data.to_csv(cleaned_file_path, index=False)
print(f"Cleaned dataset saved to {cleaned_file_path}")
