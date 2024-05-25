import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_and_preprocess_data():
    telemetry_data = pd.read_csv('../data/telemetry_data.csv')
    telemetry_data.dropna(inplace=True)
    
    # Copy the 'lap_time' column
    cleaned_telemetry_data = telemetry_data.copy()
    
    # Perform standard scaling on selected columns
    scaler = StandardScaler()
    cleaned_telemetry_data[['speed', 'rpm', 'throttle_position']] = scaler.fit_transform(cleaned_telemetry_data[['speed', 'rpm', 'throttle_position']])
    
    # Save the cleaned data to CSV
    cleaned_telemetry_data.to_csv('../data/cleaned_telemetry_data.csv', index=False)

if __name__ == "__main__":
    clean_and_preprocess_data()
