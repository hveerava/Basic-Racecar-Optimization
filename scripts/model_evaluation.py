import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model():
    # Load the data
    telemetry_data = pd.read_csv('../data/featured_telemetry_data.csv')
    feature_importance = pd.read_csv('../data/feature_importance.csv')

    # Prepare features and target variable
    X = telemetry_data.drop(columns=['lap_time'])
    y = telemetry_data['lap_time']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load the best model parameters found during model building
    best_params = {
        'n_estimators': 100,
        'max_features': None,  # Replace 'auto' with None
        'max_depth': 20,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'bootstrap': True
    }
    model = RandomForestRegressor(**best_params, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print evaluation metrics
    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2}')
    feature_importance.drop(feature_importance.tail(4).index,inplace=True)
    # Visualizing feature importance
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.savefig('../reports/visualizations/feature_importance.png')
    plt.show()

if __name__ == "__main__":
    evaluate_model()
