from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
from scripts.regression_tree import RegressionTree


class RandomForestRegressor:
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        bootstrap=True,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.estimators = None

    def fit(self, X, y, sample_weight=None):
        # convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        n_samples = X.shape[0]
        self.estimators = []

        # create N decision trees
        for i in range(self.n_estimators):
            X_bootstrap = X
            y_bootstrap = y
            # bootstrap the data
            if self.bootstrap:
                # sample with replacement
                indices = self._rng.choice(n_samples, size=n_samples, replace=True)
                X_bootstrap = X[indices]
                y_bootstrap = y[indices]

            tree_random_state = (
                None if self.random_state is None else self.random_state + i
            )

            tree = RegressionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=tree_random_state,
            )

            tree.fit(X_bootstrap, y_bootstrap)
            self.estimators.append(tree)

        return self

    def predict(self, X):
        if self.estimators is None:
            raise Exception("Model not fitted yet.")

        # get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.estimators])

        # average the predictions
        return np.mean(predictions, axis=0)

    def score(self, X, y, sample_weight=None):
        return self.polyfill.score(X, y, sample_weight=sample_weight)

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        if "random_state" in params:
            self._rng = np.random.RandomState(self.random_state)

        return self
