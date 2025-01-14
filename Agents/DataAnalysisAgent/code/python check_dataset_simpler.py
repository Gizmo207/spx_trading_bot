import pandas as pd

file_path = "C:\\Users\\inves\\OneDrive\\Documents\\Desktop\\TradingBotProject\\Agents\\DataAnalysisAgent\\data\\cleaned_dataset.csv"
data = pd.read_csv(file_path)

print("Checking for missing values...")
print(data.isnull().sum())

print("Checking for unique values in 'prediction' column...")
print(data['prediction'].unique())
