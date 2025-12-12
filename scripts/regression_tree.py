#!/usr/bin/env python3
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import time

from regression_tree_node import RegressionTreeNode

class RegressionTree:
    def __init__(self, min_samples_split=2):
        self._tree = None
        self._feature_names = list()
        
        self.min_sample_split = min_samples_split
    
    def fit(self, X, y):
        if hasattr(X, "columns"):
            self._feature_names = list(X.columns)
        else:
            # create generic names if it's already a numpy array
            self._feature_names = [f"f_{i}" for i in range(X.shape[1])]
        
        # convert to ensure fast operations
        X = np.array(X)
        y = np.array(y)
        
        self._tree = self._build_tree(X, y)
    
    def _build_tree(self, X, y):
        num_samples = X.shape[0]
        
        if num_samples < self.min_sample_split:
            leaf_value = np.mean(y)
            return RegressionTreeNode(value=leaf_value)
        
        feature_idx, threshold, X_left, y_left, X_right, y_right = self._find_best_split(X,y)
        
        left_child = self._build_tree(X_left, y_left)
        right_child = self._build_tree(X_right, y_right)

        return RegressionTreeNode(
            threshold=threshold,
            feature_index=feature_idx,
            right=right_child,
            left=left_child
        )
    
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
            
            # picking the first best feature
            if min_mse_current < best_min_mse:
                best_min_mse = min_mse_current
                
                # picking the first occurence of min MSE
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
        # ensure right column order
        if hasattr(X, "columns"):
            if set(X.columns) == set(self._feature_names):
                X = X[self._feature_names]
        
        if self._tree is None:
            raise ValueError("No regression tree trained.")
        
        X = np.array(X)
        
        predictions = np.array([self._predict_single_row(x, self._tree) for x in X])
        #print(f"predicted:", predictions)
        return predictions
    
    def _predict_single_row(self, x, node):
        # leaf node
        if node.value is not None:
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single_row(x, node.left_child)
        else:
            return self._predict_single_row(x, node.right_child)
        
    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
        
if __name__ == "__main__":
    X, y = make_regression(n_samples=100, n_features=10, noise=0.5, random_state=42)
    tree = RegressionTree(min_samples_split=4)
    tree.fit(X,y)
    tree.predict(X)

   