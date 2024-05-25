import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

def update_feature_importance():
    # Load the data
    telemetry_data = pd.read_csv('../data/featured_telemetry_data.csv')

    # Prepare features and target variable
    X = telemetry_data.drop(columns=['lap_time'])
    y = telemetry_data['lap_time']

    # Train a random forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Compute permutation importance
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)

    # Update importance values in feature_importance.csv
    feature_importance = pd.read_csv('../data/feature_importance.csv')
    for i, row in feature_importance.iterrows():
        feature = row['feature']
        if row['importance'] == 0:
            importance = perm_importance.importances_mean[X.columns.get_loc(feature)]
            feature_importance.at[i, 'importance'] = importance

    # Save the updated feature importance
    feature_importance.to_csv('../data/feature_importance.csv', index=False)

if __name__ == "__main__":
    update_feature_importance()
