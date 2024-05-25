import pandas as pd

def generate_insights():
    feature_importance = pd.read_csv('../data/feature_importance.csv')
    
    top_features = feature_importance.head(5)
    print("Top features influencing lap times (%):")
    print(top_features*100)
    
    # Additional insights and recommendations
    # These would be more domain-specific based on the data and analysis results.

if __name__ == "__main__":
    generate_insights()
