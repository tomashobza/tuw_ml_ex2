import pandas as pd
import sklearn as sk
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from xgboost import XGBRegressor
import time
import tracemalloc
from regression_tree import RegressionTree as our_regression_tree
import os

# load datasets
df_paths = [os.path.join("../data", f) for f in os.listdir("../data") if f.endswith("_processed.csv")]

# define param grids
our_grid = {
    'min_samples_split': [2, 5, 10],
    'max_depth': [None, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4]
}

sklearn_tree_grid = {
    'min_samples_split': [2, 5, 10],
    'max_depth': [None, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4]
}

sklearn_rf_grid = {
    'n_estimators': [50, 100],
    'min_samples_split': [2, 5],
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 2]
}

xgb_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1]
}

# estimators and their grids
estimators = [
    (our_regression_tree(), our_grid, "Our_RegressionTree"),
    (sk.tree.DecisionTreeRegressor(), sklearn_tree_grid, "sklearn_DecisionTree"),
    (sk.ensemble.RandomForestRegressor(), sklearn_rf_grid, "sklearn_RandomForest"),
    (XGBRegressor(), xgb_grid, "XGBoost")
]

def load_data(df_path):
    df_temp = pd.read_csv(df_path)
    X = df_temp.loc[:, df_temp.columns != "y"]
    y = df_temp["y"]
    return X, y

# store results
results = []

# run experiments
for df_path in df_paths:
    dataset_name = os.path.basename(df_path).replace("_processed.csv", "")
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")
    
    X, y = load_data(df_path)
    print(f"Shape: {X.shape}, Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    for estimator, param_grid, name in estimators:
        print(f"\n{name}:")
        print("-"*50)
        
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='neg_root_mean_squared_error',
            return_train_score=True
        )
        
        # measure CV time and memory
        tracemalloc.start()
        start_time = time.time()
        
        rmse_scores = cross_val_score(grid, X, y, cv=outer_cv, scoring='neg_root_mean_squared_error')
        mae_scores = cross_val_score(grid, X, y, cv=outer_cv, scoring='neg_mean_absolute_error')
        r2_scores = cross_val_score(grid, X, y, cv=outer_cv, scoring='r2')
        
        cv_time = time.time() - start_time
        current, cv_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # measure training time and memory
        tracemalloc.start()
        train_start = time.time()
        
        grid.fit(X, y)
        
        train_time = time.time() - train_start
        current, train_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        cv_memory_mb = cv_peak / 1024 / 1024
        train_memory_mb = train_peak / 1024 / 1024
        
        # calculate metrics
        rmse = -rmse_scores.mean()
        rmse_std = rmse_scores.std()
        mae = -mae_scores.mean()
        mae_std = mae_scores.std()
        r2 = r2_scores.mean()
        r2_std = r2_scores.std()
        nrmse = rmse / (y.max() - y.min())
        best_params = grid.best_params_
        
        print(f"  RMSE: {rmse:.4f} ± {rmse_std:.4f}")
        print(f"  MAE: {mae:.4f} ± {mae_std:.4f}")
        print(f"  R²: {r2:.4f} ± {r2_std:.4f}")
        print(f"  NRMSE: {nrmse:.4f}")
        print(f"  CV time: {cv_time:.2f}s, CV memory: {cv_memory_mb:.2f} MB")
        print(f"  Train time: {train_time:.2f}s, Train memory: {train_memory_mb:.2f} MB")
        print(f"  Best params: {best_params}")
        
        # store result
        result = {
            'dataset': dataset_name,
            'estimator': name,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'target_min': y.min(),
            'target_max': y.max(),
            'target_mean': y.mean(),
            'target_std': y.std(),
            'rmse_mean': rmse,
            'rmse_std': rmse_std,
            'mae_mean': mae,
            'mae_std': mae_std,
            'r2_mean': r2,
            'r2_std': r2_std,
            'nrmse': nrmse,
            'cv_time_seconds': cv_time,
            'cv_memory_mb': cv_memory_mb,
            'train_time_seconds': train_time,
            'train_memory_mb': train_memory_mb,
            'best_params': str(best_params),
            'inner_cv_folds': 5,
            'outer_cv_folds': 5
        }
        results.append(result)

# save to CSV
os.makedirs('../results', exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv('../results/nested_cv_results.csv', index=False)

print(f"\n{'='*70}")
print(f"Results saved to ../results/nested_cv_results.csv")
print(f"{'='*70}")