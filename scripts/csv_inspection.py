import pandas as pd

telemetry_data = pd.read_csv('../data/cleaned_telemetry_data.csv')
print(telemetry_data.columns)