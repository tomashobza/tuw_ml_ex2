#!/usr/bin/env python3
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np

from regression_tree_node import RegressionTreeNode

class RegressionTree:
    def __init__(self, min_samples_split=2):
        self._tree = RegressionTreeNode()
        self.min_sample_split = min_samples_split
    
    def fit(self, X, y):
        print("Fitting model...")
        print(self._find_best_split(X,y))
        
    
    
    
    
    # TODO add random subset of features selection ?
    # TODO selection based on MAE
    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape
        
        if num_samples < 2:
                return None, None, None, None, None, None, None
            
        best_feature_idx = None
        best_threshold = None
        best_min_mse = float('inf')
        
        for feature_idx in range(num_features):
            
            feature = X[:, feature_idx] 
            sorted_idx = np.argsort(feature)
            
            X_sorted = X[sorted_idx]
            y_sorted = y[sorted_idx]
            feature_sorted = feature[sorted_idx]

            valid = feature_sorted[1:] != feature_sorted[:-1]
            if not valid.any():
                continue
            
            y_sorted = np.array(y_sorted)
            
            cum_sum = np.cumsum(y_sorted)
            total_sum = cum_sum[-1]
            
            sums_left = cum_sum[:-1]
            sums_right = total_sum - sums_left
            
            counts_left = np.arange(1, num_samples)
            counts_right = num_samples - counts_left
            
            averages_left = sums_left / counts_left # y_i_hat (estimates)
            averages_right = sums_right / counts_right
            
            # compute MSE
            y_squared = y_sorted**2 # y_i^2 (real)
            cum_sum_squared = np.cumsum(y_squared)
            total_squared_sum = cum_sum_squared[-1]
            
            squared_sums_left = cum_sum_squared[:-1]
            squared_sums_right = total_squared_sum - squared_sums_left
            
            # sum(y_i - y_i_hat)^2 = sum(y_i^2 - y_i_hat^2) = sum(y_i^2 - n*average^2)
            sse_left = squared_sums_left - counts_left * averages_left**2
            sse_right = squared_sums_right - counts_right * averages_right**2
            
            total_mse = (sse_left + sse_right) / num_samples
            total_mse = total_mse[valid]
            
            min_mse_current = min(total_mse)
            
            if min_mse_current < best_min_mse:
                best_min_mse = min_mse_current
                min_mse_idx = np.argmin(total_mse)
                best_feature_idx = feature_idx
                
                x_prev = X_sorted[:-1, feature_idx]
                x_next = X_sorted[1:, feature_idx]
                thresholds = (x_prev + x_next) / 2.0
                thresholds = thresholds[valid]
                
                best_threshold = thresholds[min_mse_idx]
                
        if best_feature_idx is None:
            return None, None, None, None, None, None, None
        
        X_left = X[X[:, best_feature_idx] <= best_threshold]
        y_left = y[X[:, best_feature_idx] <= best_threshold]
        
        X_right = X[X[:, best_feature_idx] > best_threshold]
        y_right = y[X[:, best_feature_idx] > best_threshold]
        
        return (
            best_feature_idx, 
            best_threshold, 
            X_left, 
            y_left, 
            X_right, 
            y_right
        )
    
        
        
        
       
        
        
        
        
        
        
        

    def predict(self, X):
        print("Predicting...")
        
    def score(self, X, y):
        print("Score")
        
if __name__ == "__main__":
    X, y = make_regression(n_samples=10, n_features=1, noise=0.5, random_state=42)
    # split the dataset into training and testing sets
    # X_train, X_test = X[:500], X[500:]
    # y_train, y_test = y[:500], y[500:]
    tree = RegressionTree()
    tree.fit(X,y)