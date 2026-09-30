import numpy as np

class SVM:
    def __init__(
        self,
        C=1.0,
        kernel="linear",
        gamma=1.0,
        degree=3,
        max_iter=100,
        tol=1e-3,
        random_state=None,
    ):
        self.C = float(C)
        self.kernel = kernel
        self.gamma = float(gamma)
        self.degree = int(degree)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state

        self.classes_ = None
        self.models_ = {}
        self.rng_ = None

    def _kernel(self, X1, X2):
        if self.kernel == "linear":
            return X1 @ X2.T

        if self.kernel == "poly":
            return (self.gamma * (X1 @ X2.T) + 1.0) ** self.degree

        if self.kernel == "rbf":
            X1_norm = np.sum(X1 ** 2, axis=1, keepdims=True)
            X2_norm = np.sum(X2 ** 2, axis=1, keepdims=True)
            distance = X1_norm + X2_norm.T - 2.0 * (X1 @ X2.T)
            distance = np.maximum(distance, 0.0)
            return np.exp(-self.gamma * distance)

        raise ValueError()

    def _fit_binary(self, X, y):
        n_samples = X.shape[0]
        alpha = np.zeros(n_samples)
        b = 0.0
        K = self._kernel(X, X)

        passes = 0
        while passes < self.max_iter:
            changed = 0

            for i in range(n_samples):
                f_i = np.sum(alpha * y * K[:, i]) + b
                error_i = f_i - y[i]

                if not (y[i] * error_i < -self.tol and alpha[i] < self.C
                        or y[i] * error_i > self.tol and alpha[i] > 0):
                    continue

                candidates = np.arange(n_samples)
                candidates = candidates[candidates != i]

                j = self.rng_.choice(candidates)

                f_j = np.sum(alpha * y * K[:, j]) + b
                error_j = f_j - y[j]

                old_alpha_i = alpha[i]
                old_alpha_j = alpha[j]

                if y[i] != y[j]:
                    lower = max(0.0, alpha[j] - alpha[i])
                    upper = min(self.C, self.C + alpha[j] - alpha[i])
                else:
                    lower = max(0.0, alpha[i] + alpha[j] - self.C)
                    upper = min(self.C, alpha[i] + alpha[j])

                if lower == upper: continue

                eta = 2.0 * K[i, j] - K[i, i] - K[j, j]

                if eta >= 0: continue

                alpha[j] = old_alpha_j - y[j] * (error_i - error_j) / eta
                alpha[j] = np.clip(alpha[j], lower, upper)

                if abs(alpha[j] - old_alpha_j) < 1e-5:
                    continue

                alpha[i] = old_alpha_i + y[i] * y[j] * (old_alpha_j - alpha[j])

                b1 = (b - error_i - y[i] * (alpha[i] - old_alpha_i) * K[i, i]
                    - y[j] * (alpha[j] - old_alpha_j) * K[i, j]
                )

                b2 = (
                    b
                    - error_j
                    - y[i]
                    * (alpha[i] - old_alpha_i)
                    * K[i, j]
                    - y[j]
                    * (alpha[j] - old_alpha_j)
                    * K[j, j]
                )

                if 0 < alpha[i] < self.C:
                    b = b1
                elif 0 < alpha[j] < self.C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                changed += 1

            if changed == 0:
                passes += 1
            else:
                passes = 0

        support = alpha > 1e-5

        free_support = (
            (alpha > 1e-5)
            & (alpha < self.C - 1e-5)
        )

        if np.any(free_support):
            k = np.where(free_support)[0][0]

            b = (
                y[k]
                - np.sum(
                    alpha[support]
                    * y[support]
                    * K[support, k]
                )
            )

        return {
            "alpha": alpha[support],
            "support_vectors": X[support],
            "support_vector_labels": y[support],
            "b": b,
        }

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-dimensional array."
            )

        if y.ndim != 1:
            raise ValueError(
                "y must be a 1-dimensional array."
            )

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must have the same number of samples."
            )

        if self.C <= 0:
            raise ValueError(
                "C must be greater than 0."
            )

        if self.max_iter <= 0:
            raise ValueError(
                "max_iter must be greater than 0."
            )

        self.classes_ = np.unique(y)

        if len(self.classes_) < 2:
            raise ValueError(
                "SVM requires at least 2 classes."
            )

        self.rng_ = np.random.default_rng(
            self.random_state
        )

        self.models_ = {}

        for cls in self.classes_:
            binary_y = np.where(
                y == cls,
                1.0,
                -1.0
            )

            self.models_[cls] = self._fit_binary(
                X,
                binary_y
            )

        return self

    def decision_function(self, X):
        if self.classes_ is None:
            raise ValueError(
                "SVM has not been fitted yet."
            )

        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError(
                "X must be a 2-dimensional array."
            )

        scores = np.zeros(
            (X.shape[0], len(self.classes_))
        )

        for i, cls in enumerate(self.classes_):
            model = self.models_[cls]

            alpha = model["alpha"]
            support_vectors = model["support_vectors"]
            support_labels = model["support_vector_labels"]
            b = model["b"]

            if len(alpha) == 0:
                scores[:, i] = 0.0
                continue

            K = self._kernel(
                X,
                support_vectors
            )

            scores[:, i] = (
                K @ (alpha * support_labels)
                + b
            )

        return scores

    def predict(self, X):
        scores = self.decision_function(X)

        class_index = np.argmax(
            scores,
            axis=1
        )

        return self.classes_[class_index]

    def score(self, X, y):
        y = np.asarray(y).ravel()
        y_pred = self.predict(X)

        if y.shape != y_pred.shape:
            raise ValueError(
                "y and predictions must have the same shape."
            )

        return np.mean(y == y_pred)