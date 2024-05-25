import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def perform_eda():
    # Load the telemetry data
    telemetry_data = pd.read_csv('../data/cleaned_telemetry_data.csv')
    
    # Drop the 'lap_time' column
    telemetry_data = telemetry_data.drop(columns=['lap_time'])
    
    # Print the telemetry data (optional for debugging)
    print(telemetry_data)
    
    # Compute the correlation matrix
    correlation_matrix = telemetry_data.corr()
    
    # Plot the correlation matrix heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('../reports/visualizations/correlation_matrix.png')
    plt.show()

if __name__ == "__main__":
    perform_eda()
