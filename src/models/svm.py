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
    ):
        self.C = float(C)
        self.kernel = kernel
        self.gamma = float(gamma)
        self.degree = int(degree)
        self.max_iter = max_iter
        self.tol = tol

        self.classes_: np.ndarray | None = None
        self.models_ = {}

    def _linear_kernel(self, X1, X2):
        return np.dot(X1, X2.T)

    def _polynomial_kernel(self, X1, X2):
        return (self.gamma * np.dot(X1, X2.T) + 1.0) ** self.degree

    def _rbf_kernel(self, X1, X2):
        X1_norm_sq = np.sum(X1**2, axis=1, keepdims=True)
        X2_norm_sq = np.sum(X2**2, axis=1, keepdims=True)
        dist_sq = X1_norm_sq + X2_norm_sq.T - 2.0 * np.dot(X1, X2.T)
        return np.exp(-self.gamma * dist_sq)

    def _kernel(self, X1, X2):
        if self.kernel == "linear":
            return self._linear_kernel(X1, X2)
        elif self.kernel == "poly":
            return self._polynomial_kernel(X1, X2)
        elif self.kernel == "rbf":
            return self._rbf_kernel(X1, X2)
        else:
            raise ValueError(f"Kernel '{self.kernel}' khong hop le.")

    def _fit_binary(self, X, y):
        n_samples, n_features = X.shape
        alpha = np.zeros(n_samples)
        b = 0.0

        K = self._kernel(X, X)

        passes = 0
        while passes < self.max_iter:
            num_changed_alphas = 0

            for i in range(n_samples):
                f_xi = np.sum(alpha * y * K[:, i]) + b
                E_i = f_xi - y[i]

                if (y[i] * E_i < -self.tol and alpha[i] < self.C) or (
                    y[i] * E_i > self.tol and alpha[i] > 0
                ):
                    j = np.random.choice([idx for idx in range(n_samples) if idx != i])

                    f_xj = np.sum(alpha * y * K[:, j]) + b
                    E_j = f_xj - y[j]

                    alpha_i_old = alpha[i]
                    alpha_j_old = alpha[j]

                    if y[i] != y[j]:
                        L = max(0.0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0.0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])

                    if L == H: continue

                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0: continue

                    alpha[j] = alpha_j_old - (y[j] * (E_i - E_j)) / eta

                    if alpha[j] > H: alpha[j] = H
                    elif alpha[j] < L: alpha[j] = L

                    if abs(alpha[j] - alpha_j_old) < 1e-5: continue

                    alpha[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha[j])

                    b1 = (
                        b
                        - E_i
                        - y[i] * (alpha[i] - alpha_i_old) * K[i, i]
                        - y[j] * (alpha[j] - alpha_j_old) * K[i, j]
                    )
                    b2 = (
                        b
                        - E_j
                        - y[i] * (alpha[i] - alpha_i_old) * K[i, j]
                        - y[j] * (alpha[j] - alpha_j_old) * K[j, j]
                    )

                    if 0 < alpha[i] < self.C: b = b1
                    elif 0 < alpha[j] < self.C: b = b2
                    else: b = (b1 + b2) / 2.0

                    num_changed_alphas += 1

            if num_changed_alphas == 0: passes += 1
            else: passes = 0

        sv_indices = alpha > 1e-5
        model = {
            "alpha": alpha[sv_indices],
            "support_vectors": X[sv_indices],
            "support_vector_labels": y[sv_indices],
            "b": b,
        }
        return model

    # =========================================================
    # 3. DECISION FUNCTION
    # f(x) = sum_{sv} (alpha_k * y_k * K(x_sv, x)) + b
    # =========================================================

    def _decision_function_binary(self, X, model):
        alpha = model["alpha"]
        sv = model["support_vectors"]
        sv_y = model["support_vector_labels"]
        b = model["b"]

        if len(alpha) == 0:
            return np.zeros(X.shape[0])

        # K_test có kích thước: (n_test, n_support_vectors)
        K_test = self._kernel(X, sv)
        # Điểm quyết định: f(x) = sum_k (alpha_k * y_k * K(x, x_k)) + b
        score = np.dot(K_test, alpha * sv_y) + b
        return score

    # =========================================================
    # 4. MULTICLASS: ONE-VS-REST (OvR)
    # =========================================================

    def fit(self, X, y):
        """Huấn luyện bộ phân loại SVM đa lớp theo cơ chế One-vs-Rest."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self.models_ = {}

        for cls in self.classes_:
            # Biến đổi nhãn: class hiện tại thành +1, các class còn lại thành -1
            y_binary = np.where(y == cls, 1.0, -1.0)
            model = self._fit_binary(X, y_binary)
            self.models_[cls] = model

        return self

    def decision_function(self, X):
        """Tính điểm quyết định của từng lớp đối với từng mẫu dữ liệu."""
        if self.classes_ is None: raise ValueError()

        classes = self.classes_
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_samples = X.shape[0]
        n_classes = len(classes)
        scores = np.zeros((n_samples, n_classes))

        for idx, cls in enumerate(classes):
            scores[:, idx] = self._decision_function_binary(X, self.models_[cls])

        return scores

    def predict(self, X):
        """Dự đoán nhãn lớp có giá trị hàm quyết định lớn nhất: argmax_c f_c(x)."""
        if self.classes_ is None: raise ValueError()

        classes = self.classes_
        scores = self.decision_function(X)
        best_class_indices = np.argmax(scores, axis=1)
        return classes[best_class_indices]

    def score(self, X, y):
        """Tính độ chính xác (Accuracy)."""
        y = np.asarray(y).ravel()
        y_pred = self.predict(X)
        if len(y) != len(y_pred):
            raise ValueError()
        return np.mean(y_pred == y)