import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.model_selection import GridSearchCV

def build_model():
    telemetry_data = pd.read_csv('../data/featured_telemetry_data.csv')
    
    # Ensure lap_time is not in X
    X = telemetry_data.drop(columns=['lap_time'])
    y = telemetry_data['lap_time']
    
    # Check if the data is sufficient for splitting
    if len(X) < 2:
        raise ValueError("Not enough data to perform train-test split.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define a custom scorer to handle UndefinedMetricWarning
    def r2_scorer(y_true, y_pred):
        if len(y_true) < 2:
            return 0  # Return 0 if the number of samples is less than 2
        else:
            return r2_score(y_true, y_pred)

    # Determine the number of splits based on the size of the training set
    n_splits = min(5, len(X_train))
    
    # Use GridSearchCV for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'max_features': ['sqrt', 'log2', None],  # Remove 'auto'
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                               cv=KFold(n_splits=n_splits, shuffle=True, random_state=42), 
                               scoring=make_scorer(r2_scorer), verbose=2, n_jobs=-1)
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'Best Parameters: {grid_search.best_params_}')
    
    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': best_model.feature_importances_})
    feature_importance.sort_values(by='importance', ascending=False, inplace=True)
    feature_importance.to_csv('../data/feature_importance.csv', index=False)

if __name__ == "__main__":
    build_model()
