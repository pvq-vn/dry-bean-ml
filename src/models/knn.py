from collections import Counter
import numpy as np 

class KNN:
    def __init__(self, k=5, weights="uniform"):
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k <= 0:
            raise ValueError("k must be a positive integer")
        if weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'")

        self.k, self.weights = int(k), weights
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()

        if X.ndim != 2 or len(X) != len(y) or self.k > len(X):
            raise ValueError()

        self.X_train, self.y_train = X.copy(), y.copy()
        return self
        
    def _distance(self, X):
        if self.X_train is None:
            raise ValueError()

        X_train = self.X_train
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)
        train_sq = np.sum(X_train ** 2, axis=1, keepdims=True).T
        dist_sq = X_sq + train_sq - 2.0 * np.dot(X, X_train.T)
        dist_sq = np.maximum(dist_sq, 0.0)
        
        return np.sqrt(dist_sq)

    def _predict_one(self, distances):
        if self.y_train is None: raise ValueError()

        y_train = self.y_train
        nearest_indices = np.argsort(distances)[:self.k]
        nearest_labels = y_train[nearest_indices]

        if self.weights == "uniform":
            return Counter(nearest_labels).most_common(1)[0][0]
        else:
            nearest_distances = distances[nearest_indices]
            zero_dist_mask = nearest_distances == 0.0
            if np.any(zero_dist_mask):
                return Counter(nearest_labels[zero_dist_mask]).most_common(1)[0][0]

            weights = 1 / (nearest_distances + 1e-12)
            class_weights = {}

            for label, w in zip(nearest_labels, weights):
                class_weights[label] = class_weights.get(label, 0.0) + w

            return max(class_weights, key=class_weights.__getitem__)

    def predict(self, X):
        if self.X_train is None or self.y_train is None: raise ValueError()

        X_train = self.X_train
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != X_train.shape[1]:
            raise ValueError()

        distances = self._distance(X)

        return np.array([self._predict_one(distance) for distance in distances])

    def score(self, X, y):
        y = np.asarray(y).ravel()
        predictions = self.predict(X)
        if len(y) != len(predictions): raise ValueError()

        return np.mean(predictions == y)
