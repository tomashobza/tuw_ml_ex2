import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, cross_validate
from xgboost import XGBRegressor
import time
import tracemalloc
from regression_tree import RegressionTree as our_regression_tree
from regression_random_forest import RandomForestRegressor as our_random_forest
import os

# load datasets
df_paths = [os.path.join("data", f) for f in os.listdir("data") if f.endswith("_processed.csv")]

# define param grids
# our_tree_grid = None

our_tree_grid = {
    'min_samples_split': [2, 5, 10],
    # 'max_depth': [None, 5, 10, 20],
    # 'min_samples_leaf': [1, 2, 4]
}

# our_forest_grid = None

our_forest_grid = {
    'n_estimators': [10, 15, 20],
    # 'max_depth': [10],
    # 'min_samples_split': [5],
    # 'min_samples_leaf': [1],
    # 'max_features': ['log2'],
    # 'bootstrap': [True],
    # 'random_state': [42],
}

sklearn_tree_grid = None

sklearn_rf_grid = None

xgb_grid = None

# estimators and their grids
estimators = [
    (XGBRegressor(), xgb_grid, "XGBoost"),
    (sk.tree.DecisionTreeRegressor(), sklearn_tree_grid, "sklearn_DecisionTree"),
    (sk.ensemble.RandomForestRegressor(), sklearn_rf_grid, "sklearn_RandomForest"),
    (our_regression_tree(), our_tree_grid, "Our_RegressionTree"),
    (our_random_forest(), our_forest_grid, "Our_RandomForest"),
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
        outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        
        if param_grid is not None:
            model = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=inner_cv,
                scoring='neg_root_mean_squared_error',
                return_train_score=True
            )
        else:
            model = estimator
        
        # measure CV time and memory
        tracemalloc.start()
        start_time = time.time()
        
        cv_results = cross_validate(
            model, X, y, cv=outer_cv, 
            scoring=['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
            return_train_score=False
        )
        
        cv_time = time.time() - start_time
        current, cv_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # measure training time and memory
        tracemalloc.start()
        train_start = time.time()
        
        model.fit(X, y)
        
        train_time = time.time() - train_start
        current, train_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        cv_memory_mb = cv_peak / 1024 / 1024
        train_memory_mb = train_peak / 1024 / 1024
        
        # calculate metrics from cross_validate results
        rmse_scores = -cv_results['test_neg_root_mean_squared_error']
        mae_scores = -cv_results['test_neg_mean_absolute_error']
        r2_scores = cv_results['test_r2']
        
        rmse = rmse_scores.mean()
        rmse_std = rmse_scores.std()
        mae = mae_scores.mean()
        mae_std = mae_scores.std()
        r2 = r2_scores.mean()
        r2_std = r2_scores.std()
        nrmse = rmse / (y.max() - y.min())
        best_params = model.best_params_ if param_grid is not None else "No grid search"
        
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