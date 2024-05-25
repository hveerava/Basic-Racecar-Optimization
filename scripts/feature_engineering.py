import pandas as pd

def create_features():
    # Load the telemetry data
    telemetry_data = pd.read_csv('../data/cleaned_telemetry_data.csv')
    
    # Create a 'lap' column assuming each lap is 60 seconds long
    telemetry_data['lap'] = (telemetry_data['time'] // 60).astype(int)

    # Load the historical race data
    historical_data = pd.read_csv('../data/historical_race_data.csv')

    # Merge telemetry data with historical race data
    telemetry_data = pd.merge(telemetry_data, historical_data, on='lap', how='left')

    # Copy the 'lap_time' column from cleaned_telemetry_data.csv
    cleaned_telemetry_data = pd.read_csv('../data/cleaned_telemetry_data.csv')
    telemetry_data['lap_time'] = cleaned_telemetry_data['lap_time']

    # Ensure the 'lap_time_y' and 'pit_stop_time' are non-zero by filling NA with mean or median
    telemetry_data['lap_time_y'] = telemetry_data['lap_time_y'].fillna(telemetry_data['lap_time_y'].mean())
    telemetry_data['pit_stop_time'] = telemetry_data['pit_stop_time'].fillna(telemetry_data['pit_stop_time'].mean())
    
    # Create the acceleration feature
    telemetry_data['acceleration'] = telemetry_data['speed'].diff() / telemetry_data['time'].diff()
    telemetry_data['acceleration'] = telemetry_data['acceleration'].fillna(0)

    # Save the data with new features
    telemetry_data.to_csv('../data/featured_telemetry_data.csv', index=False)

if __name__ == "__main__":
    create_features()
